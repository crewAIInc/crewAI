"""Tests for Qdrant document payloads."""

from collections.abc import Mapping

from crewai.rag.qdrant.utils import _create_point_from_document
from crewai.rag.types import BaseRecord
import pytest


@pytest.mark.parametrize("content", ["Original document text", ""])
@pytest.mark.parametrize(
    "metadata",
    [
        {"content": "Metadata text", "source": "example.txt"},
        [{"content": "Metadata text", "source": "example.txt"}],
    ],
    ids=["mapping", "list"],
)
def test_document_content_is_not_overwritten_by_metadata(
    content: str,
    metadata: Mapping[str, str | int | float | bool]
    | list[Mapping[str, str | int | float | bool]],
) -> None:
    document: BaseRecord = {
        "content": content,
        "metadata": metadata,
    }

    point = _create_point_from_document(document, [1.0, 0.0, 0.0])

    assert point.payload == {"content": content, "source": "example.txt"}
