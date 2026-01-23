"""Integration tests for LLM multimodal functionality with cassettes.

These tests make actual API calls (recorded via VCR cassettes) to verify
multimodal content is properly sent and processed by each provider.
"""

from pathlib import Path

import pytest

from crewai.llm import LLM
from crewai_files import (
    AudioFile,
    File,
    ImageFile,
    PDFFile,
    TextFile,
    VideoFile,
    format_multimodal_content,
)
from crewai_files.resolution.resolver import FileResolver, FileResolverConfig


# Path to test data files
TEST_FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "crewai-files" / "tests" / "fixtures"
TEST_IMAGE_PATH = TEST_FIXTURES_DIR / "revenue_chart.png"
TEST_TEXT_PATH = TEST_FIXTURES_DIR / "review_guidelines.txt"
TEST_VIDEO_PATH = TEST_FIXTURES_DIR / "sample_video.mp4"
TEST_AUDIO_PATH = TEST_FIXTURES_DIR / "sample_audio.wav"


@pytest.fixture
def test_image_bytes() -> bytes:
    """Load test image bytes."""
    return TEST_IMAGE_PATH.read_bytes()


@pytest.fixture
def test_text_bytes() -> bytes:
    """Load test text bytes."""
    return TEST_TEXT_PATH.read_bytes()


@pytest.fixture
def test_video_bytes() -> bytes:
    """Load test video bytes."""
    if not TEST_VIDEO_PATH.exists():
        pytest.skip("sample_video.mp4 fixture not found")
    return TEST_VIDEO_PATH.read_bytes()


@pytest.fixture
def test_audio_bytes() -> bytes:
    """Load test audio bytes."""
    if not TEST_AUDIO_PATH.exists():
        pytest.skip("sample_audio.wav fixture not found")
    return TEST_AUDIO_PATH.read_bytes()


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


class TestOpenAIO4MiniMultimodalIntegration:
    """Integration tests for OpenAI o4-mini reasoning model with vision."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test o4-mini can describe an image."""
        llm = LLM(model="openai/o4-mini")
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


class TestOpenAIGPT41MiniMultimodalIntegration:
    """Integration tests for OpenAI GPT-4.1-mini with vision."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test GPT-4.1-mini can describe an image."""
        llm = LLM(model="openai/gpt-4.1-mini")
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


class TestOpenAIGPT5MultimodalIntegration:
    """Integration tests for OpenAI GPT-5 with vision."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test GPT-5 can describe an image."""
        llm = LLM(model="openai/gpt-5")
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


class TestOpenAIGPT5MiniMultimodalIntegration:
    """Integration tests for OpenAI GPT-5-mini with vision."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test GPT-5-mini can describe an image."""
        llm = LLM(model="openai/gpt-5-mini")
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


class TestOpenAIGPT5NanoMultimodalIntegration:
    """Integration tests for OpenAI GPT-5-nano with vision."""

    @pytest.mark.vcr()
    def test_describe_image(self, test_image_bytes: bytes) -> None:
        """Test GPT-5-nano can describe an image."""
        llm = LLM(model="openai/gpt-5-nano")
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
        llm = LLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0")
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
        llm = LLM(model="bedrock/anthropic.claude-3-haiku-20240307-v1:0")
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

    @pytest.mark.vcr()
    def test_analyze_video_file(self, test_video_bytes: bytes) -> None:
        """Test Gemini can analyze a video file."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        files = {"video": VideoFile(source=test_video_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe what you see in this video in one sentence. Be brief.",
            files,
        )

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_analyze_audio_file(self, test_audio_bytes: bytes) -> None:
        """Test Gemini can analyze an audio file."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        files = {"audio": AudioFile(source=test_audio_bytes)}

        messages = _build_multimodal_message(
            llm,
            "Describe what you hear in this audio in one sentence. Be brief.",
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


def _build_multimodal_message_with_upload(
    llm: LLM, prompt: str, files: dict
) -> tuple[list[dict], list[dict]]:
    """Build a multimodal message using file_id uploads instead of inline base64.

    Note: OpenAI Chat Completions API only supports file_id for PDFs via
    type="file", not for images. For image file_id support, OpenAI requires
    the Responses API (type="input_image"). Since crewAI uses Chat Completions,
    we test file_id uploads with Anthropic which supports file_id for all types.

    Returns:
        Tuple of (messages, content_blocks) where content_blocks can be inspected
        to verify file_id was used.
    """
    from crewai_files.formatting.anthropic import AnthropicFormatter

    config = FileResolverConfig(prefer_upload=True)
    resolver = FileResolver(config=config)
    formatter = AnthropicFormatter()

    content_blocks = []
    for file in files.values():
        resolved = resolver.resolve(file, "anthropic")
        block = formatter.format_block(file, resolved)
        if block is not None:
            content_blocks.append(block)

    messages = [
        {
            "role": "user",
            "content": [
                llm.format_text_content(prompt),
                *content_blocks,
            ],
        }
    ]
    return messages, content_blocks


def _build_responses_message_with_upload(
    llm: LLM, prompt: str, files: dict
) -> tuple[list[dict], list[dict]]:
    """Build a Responses API message using file_id uploads.

    The Responses API supports file_id for images via type="input_image".

    Returns:
        Tuple of (messages, content_blocks) where content_blocks can be inspected
        to verify file_id was used.
    """
    from crewai_files.formatting import OpenAIResponsesFormatter

    config = FileResolverConfig(prefer_upload=True)
    resolver = FileResolver(config=config)

    content_blocks = []
    for file in files.values():
        resolved = resolver.resolve(file, "openai")
        block = OpenAIResponsesFormatter.format_block(resolved, file.content_type)
        content_blocks.append(block)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                *content_blocks,
            ],
        }
    ]
    return messages, content_blocks


class TestAnthropicFileUploadIntegration:
    """Integration tests for Anthropic multimodal with file_id uploads.

    We test file_id uploads with Anthropic because OpenAI Chat Completions API
    only supports file_id references for PDFs (type="file"), not images.
    OpenAI's Responses API supports image file_id (type="input_image"), but
    crewAI currently uses Chat Completions. Anthropic supports file_id for
    all content types including images.
    """

    @pytest.mark.vcr()
    def test_describe_image_with_file_id(self, test_image_bytes: bytes) -> None:
        """Test Anthropic can describe an image uploaded via Files API."""
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022")
        files = {"image": ImageFile(source=test_image_bytes)}

        messages, content_blocks = _build_multimodal_message_with_upload(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        # Verify we're using file_id, not base64
        assert len(content_blocks) == 1
        source = content_blocks[0].get("source", {})
        assert source.get("type") == "file", (
            f"Expected source type 'file' for file_id upload, got '{source.get('type')}'. "
            "This test verifies file_id uploads work - if falling back to base64, "
            "check that the Anthropic Files API uploader is working correctly."
        )
        assert "file_id" in source, "Expected file_id in source for file_id upload"

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0


class TestOpenAIResponsesFileUploadIntegration:
    """Integration tests for OpenAI Responses API with file_id uploads.

    The Responses API supports file_id for images via type="input_image",
    unlike Chat Completions which only supports file_id for PDFs.
    """

    @pytest.mark.vcr()
    def test_describe_image_with_file_id(self, test_image_bytes: bytes) -> None:
        """Test OpenAI Responses API can describe an image uploaded via Files API."""
        llm = LLM(model="openai/gpt-4o-mini", api="responses")
        files = {"image": ImageFile(source=test_image_bytes)}

        messages, content_blocks = _build_responses_message_with_upload(
            llm,
            "Describe this image in one sentence. Be brief.",
            files,
        )

        # Verify we're using file_id with input_image type
        assert len(content_blocks) == 1
        block = content_blocks[0]
        assert block.get("type") == "input_image", (
            f"Expected type 'input_image' for Responses API, got '{block.get('type')}'. "
            "This test verifies file_id uploads work with the Responses API."
        )
        assert "file_id" in block, "Expected file_id in block for file_id upload"

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_describe_image_via_format_api(self, test_image_bytes: bytes) -> None:
        """Test format_multimodal_content with api='responses' parameter."""
        llm = LLM(model="openai/gpt-4o-mini", api="responses")
        files = {"image": ImageFile(source=test_image_bytes)}

        content_blocks = format_multimodal_content(files, "openai", api="responses")

        # Verify content blocks use Responses API format
        assert len(content_blocks) == 1
        block = content_blocks[0]
        assert block.get("type") == "input_image", (
            f"Expected type 'input_image' for Responses API, got '{block.get('type')}'"
        )
        # Should have image_url (base64 data URL) since we're not forcing upload
        assert "image_url" in block, "Expected image_url in block for inline image"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image in one sentence."},
                    *content_blocks,
                ],
            }
        ]

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.vcr()
    def test_describe_image_via_format_api_with_upload(self, test_image_bytes: bytes) -> None:
        """Test format_multimodal_content with prefer_upload=True uploads the file."""
        llm = LLM(model="openai/gpt-4o-mini", api="responses")
        files = {"image": ImageFile(source=test_image_bytes)}

        content_blocks = format_multimodal_content(
            files, "openai", api="responses", prefer_upload=True
        )

        # Verify content blocks use file_id from upload
        assert len(content_blocks) == 1
        block = content_blocks[0]
        assert block.get("type") == "input_image", (
            f"Expected type 'input_image' for Responses API, got '{block.get('type')}'"
        )
        assert "file_id" in block, (
            "Expected file_id in block when prefer_upload=True. "
            f"Got keys: {list(block.keys())}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image in one sentence."},
                    *content_blocks,
                ],
            }
        ]

        response = llm.call(messages)

        assert response
        assert isinstance(response, str)
        assert len(response) > 0