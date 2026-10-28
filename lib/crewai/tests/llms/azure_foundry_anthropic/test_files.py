import base64
from unittest.mock import patch

import pytest

pytest.importorskip("anthropic")

from crewai.llm import LLM
from crewai_files import ImageFile, PDFFile

MINIMAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
MINIMAL_PDF = b"%PDF-1.4 test content"


@pytest.fixture
def llm() -> LLM:
    return LLM(
        model="azure/claude-sonnet-4-6",
        endpoint="https://res.services.ai.azure.com",
        api_key="test-key",
    )


def _blocks(llm: LLM, files: dict) -> list[dict]:
    messages = [{"role": "user", "content": "Look", "files": files}]
    return llm._process_message_files(messages)[0]["content"]


def test_image_becomes_an_anthropic_image_block(llm: LLM) -> None:
    block = _blocks(llm, {"pic": ImageFile(source=MINIMAL_PNG)})[-1]

    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"


def test_pdf_becomes_an_anthropic_document_block(llm: LLM) -> None:
    block = _blocks(llm, {"doc": PDFFile(source=MINIMAL_PDF)})[-1]

    assert block["type"] == "document"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "application/pdf"


def test_provider_stays_azure(llm: LLM) -> None:
    assert llm.provider == "azure"


def test_large_pdf_is_inlined_and_never_uploaded(llm: LLM) -> None:
    large_pdf = MINIMAL_PDF + b" " * (6 * 1024 * 1024)

    with patch("crewai_files.resolution.resolver.get_uploader") as get_uploader:
        block = _blocks(llm, {"doc": PDFFile(source=large_pdf)})[-1]

    get_uploader.assert_not_called()
    assert block["type"] == "document"
    assert block["source"]["type"] == "base64"
