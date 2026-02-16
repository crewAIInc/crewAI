"""Unit tests for LLM multimodal functionality across all providers."""

import asyncio
import base64
import json
import logging
import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai_files import File, ImageFile, PDFFile, TextFile, format_multimodal_content

# Check for optional provider dependencies
try:
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from crewai.llms.providers.azure.completion import AzureCompletion
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

try:
    from crewai.llms.providers.bedrock.completion import BedrockCompletion
    HAS_BEDROCK = True
except ImportError:
    HAS_BEDROCK = False


# Minimal valid PNG for testing
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

MINIMAL_PDF = b"%PDF-1.4 test content"


@pytest.fixture(autouse=True)
def mock_api_keys():
    """Mock API keys for all providers."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENAI_API_KEY": "test-key",
        "GOOGLE_API_KEY": "test-key",
        "AZURE_API_KEY": "test-key",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-key",
    }
    with patch.dict(os.environ, env_vars):
        yield


class TestLiteLLMMultimodal:
    """Tests for LLM class (litellm wrapper) multimodal functionality.

    These tests use `is_litellm=True` to ensure the litellm wrapper is used
    instead of native providers.
    """

    def test_supports_multimodal_gpt4o(self) -> None:
        """Test GPT-4o model supports multimodal."""
        llm = LLM(model="gpt-4o", is_litellm=True)
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_gpt4_turbo(self) -> None:
        """Test GPT-4 Turbo model supports multimodal."""
        llm = LLM(model="gpt-4-turbo", is_litellm=True)
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_claude3(self) -> None:
        """Test Claude 3 model supports multimodal via litellm."""
        # Use litellm/ prefix to avoid native provider import
        llm = LLM(model="litellm/claude-3-sonnet-20240229")
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_gemini(self) -> None:
        """Test Gemini model supports multimodal."""
        llm = LLM(model="gemini/gemini-pro", is_litellm=True)
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_gpt35_does_not(self) -> None:
        """Test GPT-3.5 model does not support multimodal."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)
        assert llm.supports_multimodal() is False

    def test_format_multimodal_content_image(self) -> None:
        """Test formatting image content."""
        llm = LLM(model="gpt-4o", is_litellm=True)
        files = {"chart": ImageFile(source=MINIMAL_PNG)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert "data:image/png;base64," in result[0]["image_url"]["url"]

    def test_format_multimodal_content_unsupported_type(self) -> None:
        """Test unsupported content type is skipped."""
        llm = LLM(model="gpt-4o", is_litellm=True)  # OpenAI doesn't support text files
        files = {"doc": TextFile(source=b"hello world")}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert result == []


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic SDK not installed")
class TestAnthropicMultimodal:
    """Tests for Anthropic provider multimodal functionality."""

    def test_supports_multimodal_claude3(self) -> None:
        """Test Claude 3 supports multimodal."""
        llm = LLM(model="anthropic/claude-3-sonnet-20240229")
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_claude4(self) -> None:
        """Test Claude 4 supports multimodal."""
        llm = LLM(model="anthropic/claude-4-opus")
        assert llm.supports_multimodal() is True

    def test_format_multimodal_content_image(self) -> None:
        """Test Anthropic image format uses source-based structure."""
        llm = LLM(model="anthropic/claude-3-sonnet-20240229")
        files = {"chart": ImageFile(source=MINIMAL_PNG)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/png"
        assert "data" in result[0]["source"]

    def test_format_multimodal_content_pdf(self) -> None:
        """Test Anthropic PDF format uses document structure."""
        llm = LLM(model="anthropic/claude-3-sonnet-20240229")
        files = {"doc": PDFFile(source=MINIMAL_PDF)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert result[0]["type"] == "document"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "application/pdf"


class TestOpenAIMultimodal:
    """Tests for OpenAI provider multimodal functionality."""

    def test_supports_multimodal_gpt4o(self) -> None:
        """Test GPT-4o supports multimodal."""
        llm = LLM(model="openai/gpt-4o")
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_gpt4_vision(self) -> None:
        """Test GPT-4 Vision supports multimodal."""
        llm = LLM(model="openai/gpt-4-vision-preview")
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_o1(self) -> None:
        """Test O1 model supports multimodal."""
        llm = LLM(model="openai/o1-preview")
        assert llm.supports_multimodal() is True

    def test_does_not_support_gpt35(self) -> None:
        """Test GPT-3.5 does not support multimodal."""
        llm = LLM(model="openai/gpt-3.5-turbo")
        assert llm.supports_multimodal() is False

    def test_format_multimodal_content_image(self) -> None:
        """Test OpenAI uses image_url format."""
        llm = LLM(model="openai/gpt-4o")
        files = {"chart": ImageFile(source=MINIMAL_PNG)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        url = result[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Verify base64 content
        b64_data = url.split(",")[1]
        assert base64.b64decode(b64_data) == MINIMAL_PNG


class TestGeminiMultimodal:
    """Tests for Gemini provider multimodal functionality."""

    def test_supports_multimodal_always_true(self) -> None:
        """Test Gemini always supports multimodal."""
        llm = LLM(model="gemini/gemini-pro")
        assert llm.supports_multimodal() is True

    def test_format_multimodal_content_image(self) -> None:
        """Test Gemini uses inlineData format."""
        llm = LLM(model="gemini/gemini-pro")
        files = {"chart": ImageFile(source=MINIMAL_PNG)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert "inlineData" in result[0]
        assert result[0]["inlineData"]["mimeType"] == "image/png"
        assert "data" in result[0]["inlineData"]

    def test_format_text_content(self) -> None:
        """Test Gemini text format uses simple text key."""
        llm = LLM(model="gemini/gemini-pro")

        result = llm.format_text_content("Hello world")

        assert result == {"text": "Hello world"}


@pytest.mark.skipif(not HAS_AZURE, reason="Azure AI Inference SDK not installed")
class TestAzureMultimodal:
    """Tests for Azure OpenAI provider multimodal functionality."""

    @pytest.fixture(autouse=True)
    def mock_azure_env(self):
        """Mock Azure-specific environment variables."""
        env_vars = {
            "AZURE_API_KEY": "test-key",
            "AZURE_API_BASE": "https://test.openai.azure.com",
            "AZURE_API_VERSION": "2024-02-01",
        }
        with patch.dict(os.environ, env_vars):
            yield

    def test_supports_multimodal_gpt4o(self) -> None:
        """Test Azure GPT-4o supports multimodal."""
        llm = LLM(model="azure/gpt-4o")
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_gpt4_turbo(self) -> None:
        """Test Azure GPT-4 Turbo supports multimodal."""
        llm = LLM(model="azure/gpt-4-turbo")
        assert llm.supports_multimodal() is True

    def test_does_not_support_gpt35(self) -> None:
        """Test Azure GPT-3.5 does not support multimodal."""
        llm = LLM(model="azure/gpt-35-turbo")
        assert llm.supports_multimodal() is False

    def test_format_multimodal_content_image(self) -> None:
        """Test Azure uses same format as OpenAI."""
        llm = LLM(model="azure/gpt-4o")
        files = {"chart": ImageFile(source=MINIMAL_PNG)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert "data:image/png;base64," in result[0]["image_url"]["url"]


@pytest.mark.skipif(not HAS_BEDROCK, reason="AWS Bedrock SDK not installed")
class TestBedrockMultimodal:
    """Tests for AWS Bedrock provider multimodal functionality."""

    @pytest.fixture(autouse=True)
    def mock_bedrock_env(self):
        """Mock AWS-specific environment variables."""
        env_vars = {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
        with patch.dict(os.environ, env_vars):
            yield

    def test_supports_multimodal_claude3(self) -> None:
        """Test Bedrock Claude 3 supports multimodal."""
        llm = LLM(model="bedrock/anthropic.claude-3-sonnet")
        assert llm.supports_multimodal() is True

    def test_does_not_support_claude2(self) -> None:
        """Test Bedrock Claude 2 does not support multimodal."""
        llm = LLM(model="bedrock/anthropic.claude-v2")
        assert llm.supports_multimodal() is False

    def test_format_multimodal_content_image(self) -> None:
        """Test Bedrock uses Converse API image format."""
        llm = LLM(model="bedrock/anthropic.claude-3-sonnet")
        files = {"chart": ImageFile(source=MINIMAL_PNG)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert "image" in result[0]
        assert result[0]["image"]["format"] == "png"
        assert "source" in result[0]["image"]
        assert "bytes" in result[0]["image"]["source"]

    def test_format_multimodal_content_pdf(self) -> None:
        """Test Bedrock uses Converse API document format."""
        llm = LLM(model="bedrock/anthropic.claude-3-sonnet")
        files = {"doc": PDFFile(source=MINIMAL_PDF)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert "document" in result[0]
        assert result[0]["document"]["format"] == "pdf"
        assert "source" in result[0]["document"]


class TestBaseLLMMultimodal:
    """Tests for BaseLLM default multimodal behavior."""

    def test_base_supports_multimodal_false(self) -> None:
        """Test base implementation returns False."""
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model="test")
        assert llm.supports_multimodal() is False

    def test_base_format_text_content(self) -> None:
        """Test base text formatting uses OpenAI/Anthropic style."""
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model="test")
        result = llm.format_text_content("Hello")
        assert result == {"type": "text", "text": "Hello"}


class TestMultipleFilesFormatting:
    """Tests for formatting multiple files at once."""

    def test_format_multiple_images(self) -> None:
        """Test formatting multiple images."""
        llm = LLM(model="gpt-4o")
        files = {
            "chart1": ImageFile(source=MINIMAL_PNG),
            "chart2": ImageFile(source=MINIMAL_PNG),
        }

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 2

    def test_format_mixed_supported_and_unsupported(self) -> None:
        """Test only supported types are formatted."""
        llm = LLM(model="gpt-4o")  # OpenAI - images only
        files = {
            "chart": ImageFile(source=MINIMAL_PNG),
            "doc": PDFFile(source=MINIMAL_PDF),  # Not supported by OpenAI
            "text": TextFile(source=b"hello"),  # Not supported
        }

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1  # Only image supported

    def test_format_empty_files_dict(self) -> None:
        """Test empty files dict returns empty list."""
        llm = LLM(model="gpt-4o")

        result = format_multimodal_content({}, llm.model)

        assert result == []


class TestFilesStrippedWhenMultimodalUnsupported:
    """Tests that File objects are always stripped from messages before API call.

    Regression tests for https://github.com/crewAIInc/crewAI/issues/4498
    TypeError: Object of type File is not JSON serializable
    """

    def test_base_llm_strips_files_when_multimodal_not_supported(self) -> None:
        """BaseLLM._process_message_files must remove files when multimodal is off."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-no-multimodal")
        assert llm.supports_multimodal() is False

        messages = [
            {
                "role": "user",
                "content": "Describe this file",
                "files": {"doc": File(source=MINIMAL_PDF)},
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]
        assert result[0]["content"] == "Describe this file"

    def test_base_llm_strips_files_logs_warning(self, caplog) -> None:
        """BaseLLM logs a warning when dropping files on non-multimodal models."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-no-multimodal")
        messages = [
            {
                "role": "user",
                "content": "Describe this",
                "files": {"img": ImageFile(source=MINIMAL_PNG)},
            }
        ]

        with caplog.at_level(logging.WARNING):
            llm._process_message_files(messages)

        assert any("dropped from the request" in r.message for r in caplog.records)

    def test_litellm_strips_files_when_multimodal_not_supported(self) -> None:
        """LLM (litellm wrapper) strips files for non-multimodal models."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)
        assert llm.supports_multimodal() is False

        messages = [
            {
                "role": "user",
                "content": "Describe this file",
                "files": {"doc": File(source=MINIMAL_PDF)},
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]
        assert result[0]["content"] == "Describe this file"

    def test_litellm_async_strips_files_when_multimodal_not_supported(self) -> None:
        """LLM async path strips files for non-multimodal models."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)
        assert llm.supports_multimodal() is False

        messages = [
            {
                "role": "user",
                "content": "Describe this file",
                "files": {"doc": File(source=MINIMAL_PDF)},
            }
        ]

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                llm._aprocess_message_files(messages)
            )
        finally:
            loop.close()

        assert "files" not in result[0]
        assert result[0]["content"] == "Describe this file"

    def test_openai_native_strips_files_when_multimodal_not_supported(self) -> None:
        """OpenAI native provider strips files for non-vision models."""
        llm = LLM(model="openai/gpt-3.5-turbo")
        assert llm.supports_multimodal() is False

        messages = [
            {
                "role": "user",
                "content": "Describe this file",
                "files": {"doc": File(source=MINIMAL_PDF)},
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]

    def test_messages_are_json_serializable_after_file_stripping(self) -> None:
        """After _process_message_files, messages must be JSON serializable."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)

        messages = [
            {
                "role": "user",
                "content": "Analyze this",
                "files": {"my_file": File(source=MINIMAL_PDF)},
            }
        ]

        result = llm._process_message_files(messages)

        json.dumps(result)

    def test_multiple_messages_all_stripped(self) -> None:
        """All messages with files get stripped, not just the first."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)

        messages = [
            {
                "role": "user",
                "content": "First",
                "files": {"a": File(source=MINIMAL_PDF)},
            },
            {
                "role": "user",
                "content": "Second",
                "files": {"b": ImageFile(source=MINIMAL_PNG)},
            },
        ]

        result = llm._process_message_files(messages)

        for msg in result:
            assert "files" not in msg

    def test_messages_without_files_unchanged(self) -> None:
        """Messages without files are not modified."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        result = llm._process_message_files(messages)

        assert result == messages
