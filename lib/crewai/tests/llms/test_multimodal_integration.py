"""Integration tests for LLM multimodal functionality with cassettes.

These tests make actual API calls (recorded via VCR cassettes) to verify
multimodal content is properly sent and processed by each provider.
"""

from pathlib import Path

import pytest

from crewai.llm import LLM
from crewai_files import File, ImageFile, PDFFile, TextFile, format_multimodal_content


# Path to test data files
TEST_FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "crewai-files" / "tests" / "fixtures"
TEST_IMAGE_PATH = TEST_FIXTURES_DIR / "revenue_chart.png"
TEST_TEXT_PATH = TEST_FIXTURES_DIR / "review_guidelines.txt"


@pytest.fixture
def test_image_bytes() -> bytes:
    """Load test image bytes."""
    return TEST_IMAGE_PATH.read_bytes()


@pytest.fixture
def test_text_bytes() -> bytes:
    """Load test text bytes."""
    return TEST_TEXT_PATH.read_bytes()


# Minimal PDF for testing (real PDF structure)
MINIMAL_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer << /Size 4 /Root 1 0 R >>
startxref
196
%%EOF
"""


def _build_multimodal_message(llm: LLM, prompt: str, files: dict) -> list[dict]:
    """Build a multimodal message with text and file content."""
    provider = getattr(llm, "provider", None) or llm.model
    content_blocks = format_multimodal_content(files, provider)
    return [
        {
            "role": "user",
            "content": [
                llm.format_text_content(prompt),
                *content_blocks,
            ],
        }
    ]


class TestOpenAIMultimodalIntegration:
    """Integration tests for OpenAI multimodal with real API calls."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test OpenAI can describe an image."""
        llm = LLM(model="openai/gpt-4o-mini")
        files = {"image": ImageFile(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestAnthropicMultimodalIntegration:
    """Integration tests for Anthropic multimodal with real API calls."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test Anthropic can describe an image."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        files = {"image": ImageFile(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_analyze_pdf(self) -> None:
        """Test Anthropic can analyze a PDF."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        files = {"document": PDFFile(source=MINIMAL_PDF)}

        messages = _build_multimodal_message(
            llm,
            "What type of document is this? Answer in one word.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestAzureMultimodalIntegration:
    """Integration tests for Azure OpenAI multimodal with real API calls."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test Azure OpenAI can describe an image."""
        llm = LLM(model="azure/gpt-4o")
        files = {"image": ImageFile(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestBedrockMultimodalIntegration:
    """Integration tests for AWS Bedrock multimodal with real API calls."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test Bedrock Claude can describe an image."""
        llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
        files = {"image": ImageFile(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_analyze_pdf(self) -> None:
        """Test Bedrock Claude can analyze a PDF."""
        llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
        files = {"document": PDFFile(source=MINIMAL_PDF)}

        messages = _build_multimodal_message(
            llm,
            "What type of document is this? Answer in one word.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestGeminiMultimodalIntegration:
    """Integration tests for Gemini multimodal with real API calls."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test Gemini can describe an image."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        files = {"image": ImageFile(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_analyze_text_file(self, test_text_bytes: bytes) -> None:
        """Test Gemini can analyze a text file."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        files = {"readme": TextFile(source=test_text_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Summarize what this text file says in one sentence.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestLiteLLMMultimodalIntegration:
    """Integration tests for LiteLLM wrapper multimodal with real API calls."""

    @pytest.mark.vcr()
    def test_describe_image_gpt4o(self, test_image_bytes: bytes) -> None:
        """Test LiteLLM with GPT-4o can describe an image."""
        llm = LLM(model="gpt-4o-mini", is_litellm=True)
        files = {"image": ImageFile(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_describe_image_claude(self, test_image_bytes: bytes) -> None:
        """Test LiteLLM with Claude can describe an image."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022", is_litellm=True)
        files = {"image": ImageFile(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestMultipleFilesIntegration:
    """Integration tests for multiple files in a single request."""

    @pytest.mark.vcr()
    def test_multiple_images_openai(self, test_image_bytes: bytes) -> None:
        """Test OpenAI can process multiple images."""
        llm = LLM(model="openai/gpt-4o-mini")
        files = {
            "image1": ImageFile(source=test_image_bytes),
            "image2": ImageFile(source=test_image_bytes),
        }

        messages = _build_multimodal_message(
            llm,
            "How many images do you see? Answer with just the number.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert "2" in response or "two" in response.lower()

    @pytest.mark.vcr()
    def test_mixed_content_anthropic(self, test_image_bytes: bytes) -> None:
        """Test Anthropic can process image and PDF together."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        files = {
            "image": ImageFile(source=test_image_bytes),
            "document": PDFFile(source=MINIMAL_PDF),
        }

        messages = _build_multimodal_message(
            llm,
            "What types of files did I send you? List them briefly.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestGenericFileIntegration:
    """Integration tests for the generic File class with auto-detection."""

    @pytest.mark.vcr()
    def test_generic_file_image_openai(self, test_image_bytes: bytes) -> None:
        """Test generic File auto-detects image and sends correct content type."""
        llm = LLM(model="openai/gpt-4o-mini")
        files = {"image": File(source=test_image_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_generic_file_pdf_anthropic(self) -> None:
        """Test generic File auto-detects PDF and sends correct content type."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        files = {"document": File(source=MINIMAL_PDF)}

        messages = _build_multimodal_message(
            llm,
            "What type of document is this? Answer in one word.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_generic_file_text_gemini(self, test_text_bytes: bytes) -> None:
        """Test generic File auto-detects text and sends correct content type."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        files = {"content": File(source=test_text_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Summarize what this text says in one sentence.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_generic_file_mixed_types(self, test_image_bytes: bytes) -> None:
        """Test generic File works with multiple auto-detected types."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        files = {
            "chart": File(source=test_image_bytes),
            "doc": File(source=MINIMAL_PDF),
        }

        messages = _build_multimodal_message(
            llm,
            "What types of files did I send? List them briefly.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0