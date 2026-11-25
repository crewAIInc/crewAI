"""Bedrock formatting of text documents."""

from __future__ import annotations

import sys
from types import ModuleType

from crewai_files import FileBytes, TextFile
from crewai_files.formatting.api import format_multimodal_content
from crewai_files.processing.constraints import BEDROCK_CONSTRAINTS
import pytest


def _text_file(
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
    filename: str,
    content_type: str,
) -> TextFile:
    """Build a text file whose MIME type needs no system libmagic."""
    magic_module = ModuleType("magic")

    def fake_from_buffer(*args: object, **kwargs: object) -> str:
        return content_type

    magic_module.from_buffer = fake_from_buffer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "magic", magic_module)
    return TextFile(source=FileBytes(data=data, filename=filename))


@pytest.mark.parametrize(
    ("filename", "content_type", "bedrock_format"),
    [
        ("notes.txt", "text/plain", "txt"),
        ("notes.md", "text/markdown", "md"),
        ("data.csv", "text/csv", "csv"),
        ("page.html", "text/html", "html"),
    ],
)
def test_text_documents_reach_bedrock_as_document_blocks(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content_type: str,
    bedrock_format: str,
) -> None:
    """A text document Bedrock supports must not be filtered out before formatting."""
    file = _text_file(monkeypatch, b"hello", filename, content_type)

    blocks = format_multimodal_content({filename: file}, "bedrock")

    assert blocks == [
        {
            "document": {
                "name": filename,
                "format": bedrock_format,
                "source": {"bytes": b"hello"},
            }
        }
    ]


def test_bedrock_declares_text_document_constraints() -> None:
    """The provider entry mirrors the documents BedrockFormatter can emit."""
    assert BEDROCK_CONSTRAINTS.text is not None
    assert BEDROCK_CONSTRAINTS.text.max_size_bytes == 4_608_000
    assert set(BEDROCK_CONSTRAINTS.text.supported_formats) == {
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/html",
    }
