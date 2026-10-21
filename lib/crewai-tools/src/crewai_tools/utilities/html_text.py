"""Visible text of a parsed HTML document."""

from typing import Any, Final


# Elements laid out as a block of their own, so their text does not run into the text around them
_BLOCK_TAGS: Final[tuple[str, ...]] = (
    "address",
    "article",
    "aside",
    "blockquote",
    "caption",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tr",
    "ul",
)
_CELL_TAGS: Final[tuple[str, ...]] = ("td", "th")


def visible_text(soup: Any) -> str:
    """Return the text of a BeautifulSoup tree without splitting words at inline elements.

    `get_text(" ")` puts a space at every tag boundary, so `un<b>believ</b>able` reads
    `un believ able` and `<a>docs</a>.` reads `docs .`. Here text runs join as written;
    block elements and `<br>` start a new line, and table cells are set apart by a space.
    The tree is modified in place.
    """
    for line_break in soup.find_all("br"):
        line_break.replace_with("\n")
    for block in soup.find_all(_BLOCK_TAGS):
        block.insert_before("\n")
        block.insert_after("\n")
    for cell in soup.find_all(_CELL_TAGS):
        cell.insert_after(" ")
    return str(soup.get_text())
