"""Visible-text extraction for parsed HTML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final


if TYPE_CHECKING:
    from bs4 import BeautifulSoup


_BLOCK_TAGS: Final[frozenset[str]] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "caption",
        "dd",
        "details",
        "dialog",
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
        "head",
        "header",
        "hgroup",
        "hr",
        "html",
        "legend",
        "li",
        "main",
        "nav",
        "ol",
        "option",
        "p",
        "pre",
        "section",
        "summary",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "title",
        "tr",
        "ul",
    }
)
_CELL_TAGS: Final[frozenset[str]] = frozenset({"td", "th"})

_BREAK: Final = object()
_CELL_END: Final = object()


def visible_text(soup: BeautifulSoup) -> str:
    """Return the text of *soup* with words kept whole and lines kept apart.

    ``soup.get_text(" ")`` puts a space at every tag boundary, so
    ``un<b>believ</b>able`` reads ``un believ able``, ``<a>docs</a>.`` reads
    ``docs .``, and a page with no whitespace between its blocks reads as one
    line. Here text runs join exactly as written, block elements and ``<br>``
    start a new line, and table cells are separated by a space.

    The tree is walked once, without being modified, so the cost is linear in
    the size of the document. Like ``get_text()``, only plain and CDATA strings
    are read: comments, doctypes and ``<script>``/``<style>`` content are not.
    """
    # Imported here: tools that use this module keep BeautifulSoup optional.
    from bs4.element import CData, NavigableString

    text_types = (NavigableString, CData)
    parts: list[str] = []
    line_break = False
    cell_gap = False

    # Explicit stack instead of recursion, so deeply nested markup cannot
    # exhaust the interpreter stack.
    stack: list[Any] = [soup]
    while stack:
        node = stack.pop()
        if node is _BREAK:
            line_break = True
        elif node is _CELL_END:
            cell_gap = True
        elif type(node) in text_types:
            text = str(node)
            if line_break or cell_gap:
                # Whitespace next to a boundary is collapsible, so drop it.
                text = text.lstrip()
                if not text:
                    continue
                if parts:
                    if line_break:
                        while parts and not parts[-1].strip():
                            parts.pop()
                        if parts:
                            parts[-1] = parts[-1].rstrip()
                        parts.append("\n")
                    else:
                        parts.append(" ")
                line_break = cell_gap = False
            parts.append(text)
        else:
            children = getattr(node, "contents", None)
            if children is None:
                continue
            name = getattr(node, "name", None)
            if name in _BLOCK_TAGS:
                stack.append(_BREAK)
                stack.extend(reversed(children))
                stack.append(_BREAK)
            elif name in _CELL_TAGS:
                stack.append(_CELL_END)
                stack.extend(reversed(children))
            else:
                stack.extend(reversed(children))
    return "".join(parts)
