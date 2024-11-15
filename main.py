import logging
from argparse import ArgumentParser
from pathlib import Path
from ebooklib import epub, ITEM_DOCUMENT, ITEM_NAVIGATION
from ebooklib.epub import EpubBook, EpubHtml
import json
import trafilatura


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
        "--output",
        "-o",
        type=Path,
        help="Path to output file. If not provided, output will be printed to stdout",
        required=False,
    )
    args = parser.parse_args()
    epub_path = args.epub

    # these imports are slow, only run them after argument parsing in case user only runs help
    from haystack_integrations.components.generators.google_ai import (
        GoogleAIGeminiGenerator,
    )

    llm = GoogleAIGeminiGenerator(model="gemini-1.5-flash")

    book = epub.read_epub(epub_path)
    id2doc = {doc.get_id(): doc for doc in book.get_items_of_type(ITEM_DOCUMENT)}
    ordered_items = [id2doc[x[0]] for x in book.spine]
    nav = next(book.get_items_of_type(ITEM_NAVIGATION))

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

    Extract document names coresponding to each chapter, ignore non-content section such as cover, insert, afterword,... The result must be in json format: [{{"name": "chapter_1_name", "documents": ["document_1", "document_2", ...]}}, {{"name": "chapter_2_name", "documents": ["document_1", "document_2", ...]}}, ...].
    Do not use fence code for the result.
    """
    extract_chapters_res = llm.run(parts=[extract_chapters_prompt])
    chapters = json.loads(extract_chapters_res["replies"][0])
    logging.debug(f"Extracted chapters: {chapters}")

    chapter_names = [c["name"] for c in chapters]
    chapter_contents = [get_content(book, c["documents"]) for c in chapters]
    summarized_chapters = []
    for name, c in zip(chapter_names, chapter_contents):
        logging.info(f"Summarizing chapter '{name}'")
        res = llm.run(
            parts=[
                "Below is a chapter from a story, summarize it in up to 500 words.\n\n"
                + c
            ]
        )
        summarized_chapters.append(res["replies"][0])

    summary = ""
    for name, content in zip(chapter_names, summarized_chapters):
        summary += f"## {name}\n\n{content}\n\n"

    if parser.output is not None:
        with open(parser.output, "w") as f:
            f.write(summary)
    else:
        print("--------------------")
        print("# Summary")
        print(summary)


if __name__ == "__main__":
    main()
