"""Tests for the Bedrock Converse content block formatter."""

import re

from crewai_files import PDFFile
from crewai_files.core.resolved import FileReference, InlineBytes
from crewai_files.formatting.bedrock import BedrockFormatter
import pytest


MINIMAL_PDF = b"%PDF-1.4 test content"

# DocumentBlock.name constraints from the Bedrock Converse API reference.
_VALID_NAME = re.compile(r"[A-Za-z0-9()\[\]\-]+( [A-Za-z0-9()\[\]\-]+)*")


def _document_name(name: str | None) -> str:
    file = PDFFile(source=MINIMAL_PDF)
    resolved = InlineBytes(content_type="application/pdf", data=MINIMAL_PDF)
    block = BedrockFormatter().format_block(file, resolved, name=name)
    assert block is not None
    return block["document"]["name"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("report", "report"),
        ("Q3 report (final) [v2]-draft", "Q3 report (final) [v2]-draft"),
        ("q3_report.pdf", "q3 report pdf"),
        ("file_0", "file 0"),
        ("a  b\t\nc", "a b c"),
        ("  padded  ", "padded"),
        ("___", "document"),
        ("", "document"),
        (None, "document"),
    ],
)
def test_document_name_is_sanitized(name: str | None, expected: str) -> None:
    result = _document_name(name)

    assert result == expected
    assert _VALID_NAME.fullmatch(result)


def test_document_name_is_truncated_to_200_characters() -> None:
    result = _document_name("x" * 199 + "_" + "y" * 50)

    assert len(result) <= 200
    assert result == "x" * 199
    assert _VALID_NAME.fullmatch(result)


def test_s3_document_name_is_sanitized() -> None:
    file = PDFFile(source=MINIMAL_PDF)
    resolved = FileReference(
        content_type="application/pdf",
        file_id="key",
        provider="bedrock",
        file_uri="s3://bucket/key",
    )

    block = BedrockFormatter().format_block(file, resolved, name="annual_report.pdf")

    assert block is not None
    assert block["document"]["name"] == "annual report pdf"
    assert block["document"]["source"] == {"s3Location": {"uri": "s3://bucket/key"}}
