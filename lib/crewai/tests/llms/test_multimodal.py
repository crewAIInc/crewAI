"""Unit tests for LLM multimodal functionality across all providers."""

import base64
import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai_files import ImageFile, PDFFile, TextFile, format_multimodal_content

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


class TestTextFileInliningNonMultimodal:
    """Tests for text file inlining on non-multimodal models (issue #5137).

    When a model does not support multimodal input, text files should be
    inlined as plain text in the message content rather than raising a
    ValueError.
    """

    # --- BaseLLM (native provider path) ---

    def test_base_text_file_inlined_on_non_multimodal(self) -> None:
        """TextFile content is inlined when model is not multimodal (BaseLLM)."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-model")
        assert llm.supports_multimodal() is False

        text_content = b"Hello from a text file!"
        messages = [
            {
                "role": "user",
                "content": "Analyse this file",
                "files": {"readme": TextFile(source=text_content)},
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]
        assert "Hello from a text file!" in result[0]["content"]
        assert "readme" in result[0]["content"]

    def test_base_multiple_text_files_inlined(self) -> None:
        """Multiple text files are all inlined on non-multimodal model."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-model")

        messages = [
            {
                "role": "user",
                "content": "Analyse these files",
                "files": {
                    "file1": TextFile(source=b"Content of file 1"),
                    "file2": TextFile(source=b"Content of file 2"),
                },
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]
        assert "Content of file 1" in result[0]["content"]
        assert "Content of file 2" in result[0]["content"]

    def test_base_image_file_still_rejected_on_non_multimodal(self) -> None:
        """ImageFile still raises ValueError on non-multimodal model."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-model")

        messages = [
            {
                "role": "user",
                "content": "Describe this image",
                "files": {"photo": ImageFile(source=MINIMAL_PNG)},
            }
        ]

        with pytest.raises(ValueError, match="non-text files"):
            llm._process_message_files(messages)

    def test_base_mixed_text_and_image_rejects_but_inlines_text(self) -> None:
        """Mixed text+image: text is inlined, but error is raised for image."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-model")

        messages = [
            {
                "role": "user",
                "content": "Process these",
                "files": {
                    "readme": TextFile(source=b"Some text content"),
                    "photo": ImageFile(source=MINIMAL_PNG),
                },
            }
        ]

        with pytest.raises(ValueError, match="non-text files"):
            llm._process_message_files(messages)

        # Text file should have been inlined before the error
        assert "Some text content" in messages[0]["content"]

    def test_base_no_files_no_error(self) -> None:
        """Messages without files pass through unchanged."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-model")

        messages = [
            {"role": "user", "content": "No files here"},
        ]

        result = llm._process_message_files(messages)
        assert result[0]["content"] == "No files here"

    def test_base_text_file_with_empty_existing_content(self) -> None:
        """TextFile inlined when existing content is empty string."""
        from crewai.llms.base_llm import BaseLLM

        class NonMultimodalLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = NonMultimodalLLM(model="test-model")

        messages = [
            {
                "role": "user",
                "content": "",
                "files": {"doc": TextFile(source=b"File content here")},
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]
        assert "File content here" in result[0]["content"]
        # Should not start with newlines when existing content is empty
        assert not result[0]["content"].startswith("\n")

    # --- LiteLLM LLM class ---

    def test_litellm_text_file_inlined_on_non_multimodal(self) -> None:
        """TextFile content is inlined when litellm model is not multimodal."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)
        assert llm.supports_multimodal() is False

        messages = [
            {
                "role": "user",
                "content": "Analyse this file",
                "files": {"readme": TextFile(source=b"Hello from litellm test")},
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]
        assert "Hello from litellm test" in result[0]["content"]

    def test_litellm_image_file_rejected_on_non_multimodal(self) -> None:
        """ImageFile raises ValueError on non-multimodal litellm model."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)
        assert llm.supports_multimodal() is False

        messages = [
            {
                "role": "user",
                "content": "Describe this",
                "files": {"photo": ImageFile(source=MINIMAL_PNG)},
            }
        ]

        with pytest.raises(ValueError, match="non-text files"):
            llm._process_message_files(messages)

    def test_litellm_json_file_inlined_on_non_multimodal(self) -> None:
        """JSON file (application/json) is treated as text and inlined."""
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True)
        assert llm.supports_multimodal() is False

        json_content = b'{"key": "value"}'
        messages = [
            {
                "role": "user",
                "content": "Parse this JSON",
                "files": {"data": TextFile(source=json_content)},
            }
        ]

        result = llm._process_message_files(messages)

        assert "files" not in result[0]
        assert '{"key": "value"}' in result[0]["content"]

    # --- _is_text_file helper ---

    def test_is_text_file_with_text_file_instance(self) -> None:
        """_is_text_file returns True for TextFile instances."""
        from crewai.llms.base_llm import BaseLLM

        assert BaseLLM._is_text_file(TextFile(source=b"hello")) is True

    def test_is_text_file_with_image_file_instance(self) -> None:
        """_is_text_file returns False for ImageFile instances."""
        from crewai.llms.base_llm import BaseLLM

        assert BaseLLM._is_text_file(ImageFile(source=MINIMAL_PNG)) is False

    def test_is_text_file_with_pdf_file_instance(self) -> None:
        """_is_text_file returns False for PDFFile instances."""
        from crewai.llms.base_llm import BaseLLM

        assert BaseLLM._is_text_file(PDFFile(source=MINIMAL_PDF)) is False

    def test_is_text_file_with_text_content_type(self) -> None:
        """_is_text_file returns True for objects with text/* content_type."""
        from crewai.llms.base_llm import BaseLLM

        class MockFile:
            content_type = "text/plain"

        assert BaseLLM._is_text_file(MockFile()) is True

    def test_is_text_file_with_json_content_type(self) -> None:
        """_is_text_file returns True for application/json content_type."""
        from crewai.llms.base_llm import BaseLLM

        class MockFile:
            content_type = "application/json"

        assert BaseLLM._is_text_file(MockFile()) is True

    def test_is_text_file_with_xml_content_type(self) -> None:
        """_is_text_file returns True for application/xml content_type."""
        from crewai.llms.base_llm import BaseLLM

        class MockFile:
            content_type = "application/xml"

        assert BaseLLM._is_text_file(MockFile()) is True

    def test_is_text_file_with_yaml_content_type(self) -> None:
        """_is_text_file returns True for application/x-yaml content_type."""
        from crewai.llms.base_llm import BaseLLM

        class MockFile:
            content_type = "application/x-yaml"

        assert BaseLLM._is_text_file(MockFile()) is True

    def test_is_text_file_with_image_content_type(self) -> None:
        """_is_text_file returns False for image/* content_type."""
        from crewai.llms.base_llm import BaseLLM

        class MockFile:
            content_type = "image/png"

        assert BaseLLM._is_text_file(MockFile()) is False


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
