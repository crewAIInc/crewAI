"""Resolution of the content items accepted by ``RagTool.add``.

A content item is either a plain reference (a path, a URL or a piece of
text) or a document dictionary. A document dictionary carries the document
body in ``content`` while ``source`` only names where that body came from.

The helpers in this module turn such an item into the label that describes
it and the reference the loaders read, so that the RAG adapter and the
validation performed by ``RagTool.add`` always agree on which field of the
item is read from disk or fetched from the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crewai_tools.rag.data_types import DataType


@dataclass(frozen=True)
class ResolvedContentItem:
    """A content item split into the label describing it and what to load.

    Attributes:
        label: The source label, kept as the ``source`` metadata.
        reference: The string the loaders read.
        inline: True when the reference is a document body that is indexed as
            plain text instead of being read from disk or fetched.
        labelled: True when the item carried an explicit ``source`` label.
        field: The key of the document dict the reference came from, or None
            when the item is a plain reference.
    """

    label: str
    reference: str
    inline: bool
    labelled: bool
    field: str | None = None


def resolve_content_item(
    arg: str | Path | dict[str, Any], data_type: DataType | None
) -> ResolvedContentItem:
    """Resolve a content item into the label to keep and the text to index.

    A document dictionary carries the document body in ``content`` while
    ``source`` only names where that body came from. The body is therefore
    the document: it is indexed as plain text and is never read from disk and
    never fetched, so a body that looks like a path or a URL is indexed as
    the text it is. The source is kept as the ``source`` metadata.

    A source that points at a document is only loaded when the caller asks
    for a loader explicitly, or when the item carries no body at all.

    Args:
        arg: A content item, either a plain reference or a document dict.
        data_type: The data type the caller asked for, if any.

    Returns:
        The resolved item.
    """
    if not isinstance(arg, dict):
        return ResolvedContentItem(
            label=str(arg), reference=str(arg), inline=False, labelled=False
        )

    source = arg.get("source")
    content = arg.get("content")
    label = "" if source is None else str(source)
    body = content if isinstance(content, str) and content.strip() else ""

    if not body:
        # No inline body: the source, or the content on its own, is the
        # reference the loaders read.
        if source is not None:
            reference = label
            field: str | None = "source"
        else:
            reference = "" if content is None else str(content)
            field = "content"
        return ResolvedContentItem(
            label=reference,
            reference=reference,
            inline=False,
            labelled=source is not None,
            field=field,
        )

    if source is None:
        # Only a body: it is the reference, as it has always been. With an
        # explicit text type it is indexed verbatim instead of being read.
        return ResolvedContentItem(
            label=body,
            reference=body,
            inline=data_type == DataType.TEXT,
            labelled=False,
            field="content",
        )

    if data_type is None or data_type == DataType.TEXT:
        # The body is the document; the source only labels it.
        return ResolvedContentItem(
            label=label, reference=body, inline=True, labelled=True, field="content"
        )

    # An explicit loader type: the source is the document to load.
    return ResolvedContentItem(
        label=label, reference=label, inline=False, labelled=True, field="source"
    )


def parse_data_type(raw_data_type: DataType | str | None) -> DataType | None:
    """Read the data type the caller asked for.

    Args:
        raw_data_type: The ``data_type`` argument, as a enum value or a string.

    Returns:
        The requested data type, or None when the type has to be detected.

    Raises:
        ValueError: If the data type is not a known one.
    """
    if raw_data_type is None:
        return None
    if isinstance(raw_data_type, DataType):
        return None if raw_data_type == DataType.FILE else raw_data_type
    if raw_data_type == "file":
        return None
    try:
        return DataType(raw_data_type)
    except ValueError:
        raise ValueError(
            f"Invalid data_type: '{raw_data_type}'. "
            f"Valid values are: 'file' (auto-detect), or one of: "
            f"{', '.join(dt.value for dt in DataType)}"
        ) from None
