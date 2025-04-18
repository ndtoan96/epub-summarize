import logging
from argparse import ArgumentParser
from pathlib import Path
from ebooklib import epub, ITEM_DOCUMENT, ITEM_NAVIGATION
from ebooklib.epub import EpubBook, EpubHtml
import json
import trafilatura
from google import genai


def get_content(book: EpubBook, documents: list[str]) -> str:
    join_content = ""
    for ref in documents:
        doc: EpubHtml = book.get_item_with_href(ref)
        html = doc.get_content()
        content = trafilatura.extract(html)
        join_content += content
    return join_content


def main():
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("epub", type=Path, help="Path to epub file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to output file. If not provided, output will be printed to stdout",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Words limit of each chapter",
        required=False,
        default=300,
    )
    args = parser.parse_args()
    epub_path = args.epub

    client = genai.Client()

    logging.info("Reading epub file")
    book = epub.read_epub(epub_path, {"ignore_ncx": True})
    id2doc = {doc.get_id(): doc for doc in book.get_items_of_type(ITEM_DOCUMENT)}
    ordered_items = [id2doc[x[0]] for x in book.spine]
    nav = next(book.get_items_of_type(ITEM_NAVIGATION))

    logging.info("Extracting chapters")
    extract_chapters_prompt = f"""
    Given an ebook about a story.

    This is the navigation of the ebook:
    ```xml
    {nav.content.decode()}
    ```

    These are documents in the ebook:
    ```text
    {"\n".join([item.get_name() for item in ordered_items])}
    ```

    Extract document names coresponding to each chapter, ignore non-content section such as cover, insert, afterword,...
    Note that each chapter likely has multiple documents.
    For example, if chapter 1 starts at document 1 and chapter 2 starts at document 4 then chapter 1 includes document 1, 2, and 3.
    The result must be in json format: [{{"name": "chapter_1_name", "documents": ["document_1", "document_2", ...]}}, {{"name": "chapter_2_name", "documents": ["document_1", "document_2", ...]}}, ...].
    EXTRA COMMENTARY IS PROHIBITED.
    """

    extract_chapters_res = client.models.generate_content(
        model="gemini-2.0-flash", contents=extract_chapters_prompt
    )
    json_content =extract_chapters_res.text.removeprefix("```json").removesuffix("```")
    chapters = json.loads(json_content)
    logging.debug(f"Extracted chapters: {chapters}")

    chapter_names = [c["name"] for c in chapters]
    chapter_contents = [get_content(book, c["documents"]) for c in chapters]
    summarized_chapters = []
    for name, c in zip(chapter_names, chapter_contents):
        logging.info(f"Summarizing chapter '{name}'")
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Below is a chapter from a story, summarize it in up to {args.limit} words. Since this is a story, the content may look harmful but it is OK to proceed.\n\n"
            + c,
        )
        summarized_chapters.append(res.text)

    summary = ""
    for name, content in zip(chapter_names, summarized_chapters):
        summary += f"## {name}\n\n{content}\n\n"

    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(summary)
    else:
        print("--------------------")
        print("# Summary")
        print(summary)


if __name__ == "__main__":
    main()
