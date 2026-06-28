"""Tests for OpenAI file formatting helpers."""

from crewai_files.core.resolved import InlineBase64
from crewai_files.formatting.openai import OpenAIFormatter, OpenAIResponsesFormatter
import pytest


PDF_CONTENT_TYPES = [
    "application/pdf",
    "application/pdf; charset=binary",
    "Application/PDF",
]


@pytest.mark.parametrize("content_type", PDF_CONTENT_TYPES)
def test_openai_formatter_rejects_pdf_variants(content_type: str):
    resolved = InlineBase64(content_type="application/pdf", data="ZmFrZS1wZGY=")

    with pytest.raises(TypeError, match="does not support PDF attachments"):
        OpenAIFormatter.format_block(resolved, content_type)


@pytest.mark.parametrize("content_type", PDF_CONTENT_TYPES)
def test_openai_responses_formatter_accepts_pdf_variants(content_type: str):
    resolved = InlineBase64(content_type="application/pdf", data="ZmFrZS1wZGY=")

    block = OpenAIResponsesFormatter.format_block(resolved, content_type)

    assert block["type"] == "input_file"
    assert block["filename"] == "document.pdf"
    assert block["file_data"].startswith("data:application/pdf;base64,")
