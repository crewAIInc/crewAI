"""Unit tests for LLM multimodal functionality across all providers."""

import base64
import copy
import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai_files import ImageFile, PDFFile, TextFile, format_multimodal_content

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

    def test_supports_vision_override_survives_copy(self) -> None:
        llm = LLM(model="gpt-3.5-turbo", is_litellm=True, supports_vision=True)

        assert copy.copy(llm).supports_multimodal() is True
        assert copy.deepcopy(llm).supports_multimodal() is True

    def test_supports_vision_false_override_survives_copy(self) -> None:
        llm = LLM(model="gpt-4o", is_litellm=True, supports_vision=False)

        assert copy.copy(llm).supports_multimodal() is False
        assert copy.deepcopy(llm).supports_multimodal() is False

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

    def test_format_responses_pdf_with_concrete_gpt_model(self) -> None:
        """Test OpenAI Responses PDF support with an inferred GPT provider."""
        files = {"doc": PDFFile(source=MINIMAL_PDF)}

        result = format_multimodal_content(files, "gpt-4o-mini", api="responses")

        assert len(result) == 1
        assert result[0]["type"] == "input_file"
        assert result[0]["file_data"].startswith("data:application/pdf;base64,")


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

    def test_supports_vision_override_applies_to_every_provider(self) -> None:
        llm = LLM(model="openai/gpt-3.5-turbo", supports_vision=True)

        assert llm.supports_multimodal() is True

    def test_format_multimodal_content_image(self) -> None:
        """Test OpenAI uses image_url format."""
        llm = LLM(model="openai/gpt-4o")
        files = {"chart": ImageFile(source=MINIMAL_PNG)}

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        url = result[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
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

    def test_supports_multimodal_gpt41(self) -> None:
        """Test Azure GPT-4.1 supports multimodal from the catalog."""
        llm = LLM(model="azure/gpt-4.1")
        assert llm.supports_multimodal() is True

    def test_supports_multimodal_gpt5(self) -> None:
        """Test Azure GPT-5 supports multimodal from the catalog."""
        llm = LLM(model="azure/gpt-5")
        assert llm.supports_multimodal() is True

    def test_does_not_support_gpt35(self) -> None:
        """Test Azure GPT-3.5 does not support multimodal."""
        llm = LLM(model="azure/gpt-35-turbo")
        assert llm.supports_multimodal() is False

    @pytest.mark.parametrize("model", ["azure/o1-mini", "azure/o3-mini"])
    def test_does_not_support_text_only_reasoning_models(self, model: str) -> None:
        """Text-only o-series models stay rejected even though o1/o3 are vision."""
        llm = LLM(model=model)
        assert llm.supports_multimodal() is False

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("azure/gpt-4.1-2025-04-14", True),
            ("azure/gpt-4o-2024-08-06", True),
            ("azure/gpt-5-mini", True),
            ("azure/o3-mini-2025-01-31", False),
            ("azure/o1-mini-2024-09-12", False),
            ("azure/gpt-35-turbo-0613", False),
        ],
    )
    def test_versioned_deployment_names(self, model: str, expected: bool) -> None:
        assert LLM(model=model).supports_multimodal() is expected

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("azure/GPT-4.1", True),
            ("azure/GPT-5", True),
            ("azure/GPT-4o", True),
            ("azure/O3-MINI", False),
        ],
    )
    def test_deployment_names_are_case_insensitive(
        self, model: str, expected: bool
    ) -> None:
        assert LLM(model=model).supports_multimodal() is expected

    @pytest.mark.parametrize(
        "model", ["azure/claude-sonnet-4-6", "azure/azure_ai/claude-sonnet-4-6"]
    )
    def test_azure_ai_catalog_model_supports_multimodal(self, model: str) -> None:
        """Azure consults the Azure AI catalog entries for model capabilities."""
        assert LLM(model=model).supports_multimodal() is True

    def test_claude_deployment_accepted_with_override(self) -> None:
        llm = LLM(model="azure/claude-sonnet-4-6", supports_vision=True)

        assert llm.supports_multimodal() is True

    def test_double_provider_prefix_still_resolves(self) -> None:
        assert LLM(model="azure/azure/gpt-4.1").supports_multimodal() is True

    @pytest.mark.parametrize("model", ["azure/ ", "azure///", "azure/azure/"])
    def test_unusual_model_names_do_not_raise(self, model: str) -> None:
        assert LLM(model=model).supports_multimodal() is False

    def test_supports_vision_override_rejects_legacy_vision_name(self) -> None:
        llm = LLM(model="azure/gpt-4o", supports_vision=False)

        assert llm.supports_multimodal() is False

    def test_supports_vision_override_allows_text_only_model(self) -> None:
        llm = LLM(model="azure/o1-mini", supports_vision=True)

        assert llm.supports_multimodal() is True

    @pytest.mark.parametrize(
        ("model", "expected"),
        [("azure/gpt-4.1", True), ("azure/gpt-35-turbo", False)],
    )
    def test_supports_vision_none_matches_unset(self, model: str, expected: bool) -> None:
        explicit = LLM(model=model, supports_vision=None)

        assert explicit.supports_multimodal() is LLM(model=model).supports_multimodal()
        assert explicit.supports_multimodal() is expected

    @pytest.mark.parametrize("override", [True, False])
    def test_supports_vision_override_survives_copy(self, override: bool) -> None:
        llm = LLM(model="azure/vision-production", supports_vision=override)

        assert copy.copy(llm).supports_multimodal() is override
        assert copy.deepcopy(llm).supports_multimodal() is override

    def test_supports_vision_override_from_config_dict(self) -> None:
        from crewai.utilities.llm_utils import create_llm

        llm = create_llm({"model": "azure/vision-production", "supports_vision": True})

        assert llm.supports_multimodal() is True

    def test_supports_vision_override_allows_custom_deployment_name(self) -> None:
        llm = LLM(model="azure/vision-production", supports_vision=True)

        assert llm.supports_multimodal() is True

    def test_supports_vision_override_rejects_known_vision_model(self) -> None:
        llm = LLM(model="azure/gpt-4.1", supports_vision=False)

        assert llm.supports_multimodal() is False

    def test_files_accepted_when_override_enables_custom_deployment(self) -> None:
        llm = LLM(model="azure/vision-production", supports_vision=True)
        messages = [
            {"role": "user", "content": "Describe", "files": {"chart": ImageFile(source=MINIMAL_PNG)}}
        ]

        result = llm._process_message_files(messages)

        assert result[0]["content"][-1]["type"] == "image_url"

    @pytest.mark.parametrize("model", ["azure/o1-mini", "azure/o3-mini"])
    def test_files_rejected_for_text_only_models(self, model: str) -> None:
        llm = LLM(model=model)
        messages = [
            {"role": "user", "content": "Describe", "files": {"chart": ImageFile(source=MINIMAL_PNG)}}
        ]

        with pytest.raises(ValueError, match="does not support multimodal input"):
            llm._process_message_files(messages)

    def test_pdf_rejected_instead_of_silently_dropped(self) -> None:
        llm = LLM(model="azure/gpt-4.1")
        messages = [
            {"role": "user", "content": "Read", "files": {"doc": PDFFile(source=MINIMAL_PDF)}}
        ]

        with pytest.raises(ValueError, match="'doc' is application/pdf"):
            llm._process_message_files(messages)

    def test_text_file_rejected_instead_of_silently_dropped(self) -> None:
        llm = LLM(model="azure/gpt-4.1")
        messages = [
            {
                "role": "user",
                "content": "Read",
                "files": {"data": TextFile(source=b"a,b\n1,2", filename="data.csv")},
            }
        ]

        with pytest.raises(ValueError, match="'data' is text/"):
            llm._process_message_files(messages)

    def test_pdf_rejected_even_when_override_enables_vision(self) -> None:
        llm = LLM(model="azure/vision-production", supports_vision=True)
        messages = [
            {"role": "user", "content": "Read", "files": {"doc": PDFFile(source=MINIMAL_PDF)}}
        ]

        with pytest.raises(ValueError, match="accepts only"):
            llm._process_message_files(messages)

    def test_pdf_accepted_by_responses_api(self) -> None:
        llm = LLM(model="azure/gpt-4.1", api="responses")
        messages = [
            {"role": "user", "content": "Read", "files": {"doc": PDFFile(source=MINIMAL_PDF)}}
        ]

        result = llm._process_message_files(messages)

        assert result[0]["content"][-1]["type"] == "input_file"

    def test_image_accepted_by_responses_api(self) -> None:
        llm = LLM(model="azure/gpt-4.1", api="responses")
        messages = [
            {
                "role": "user",
                "content": "Look",
                "files": {"pic": ImageFile(source=MINIMAL_PNG)},
            }
        ]

        result = llm._process_message_files(messages)

        assert result[0]["content"][-1]["type"] == "input_image"

    def test_image_still_accepted_alongside_validation(self) -> None:
        llm = LLM(model="azure/gpt-4.1")
        messages = [
            {"role": "user", "content": "Look", "files": {"pic": ImageFile(source=MINIMAL_PNG)}}
        ]

        result = llm._process_message_files(messages)

        assert result[0]["content"][-1]["type"] == "image_url"

    def test_files_rejected_when_override_disables_vision_model(self) -> None:
        llm = LLM(model="azure/gpt-4.1", supports_vision=False)
        messages = [
            {"role": "user", "content": "Describe", "files": {"chart": ImageFile(source=MINIMAL_PNG)}}
        ]

        with pytest.raises(ValueError, match="does not support multimodal input"):
            llm._process_message_files(messages)

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

    @pytest.mark.parametrize(
        "model",
        [
            "bedrock/openai.gpt-6-sol",
            "bedrock/openai.gpt-6-luna",
            "bedrock/openai.gpt-6-astra",
            "bedrock/us.openai.gpt-6-sol",
            "bedrock/us.openai.gpt-6-luna",
            "bedrock/us.openai.gpt-6-astra",
            "bedrock/global.openai.gpt-6-sol",
            "bedrock/global.openai.gpt-6-luna",
            "bedrock/global.openai.gpt-6-astra",
        ],
    )
    def test_supports_multimodal_openai_gpt_6(self, model: str) -> None:
        """Test Bedrock OpenAI GPT-6 Sol/Luna/Astra support multimodal."""
        llm = LLM(model=model)
        assert llm.supports_multimodal() is True

    def test_does_not_support_gpt_oss(self) -> None:
        """Test Bedrock gpt-oss (no image input) does not support multimodal."""
        llm = LLM(model="bedrock/openai.gpt-oss-120b-1:0")
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

    def test_model_capability_uses_provider_catalog_entry(self) -> None:
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model="gpt-4.1", provider="azure")
        assert (
            llm.get_model_capability("supports_vision", providers=["azure"])
            is True
        )

    def test_model_capability_without_provider_uses_exact_model_key(self) -> None:
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model="azure/gpt-4.1")
        assert llm.get_model_capability("supports_vision") is True

    def test_model_capability_checks_multiple_provider_catalog_entries(self) -> None:
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model="claude-sonnet-4-6")
        assert (
            llm.get_model_capability(
                "supports_vision", providers=["azure", "azure_ai"]
            )
            is True
        )

    def test_model_capability_lookup_is_case_insensitive(self) -> None:
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model="GPT-4.1", provider="azure")
        assert llm.get_model_capability("supports_vision", providers=["azure"]) is True

    @pytest.mark.parametrize("model", [" ", "///", "a/b/c"])
    def test_model_capability_unusual_names_return_none(self, model: str) -> None:
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model=model, provider="azure")
        assert llm.get_model_capability("supports_vision", providers=["azure"]) is None

    def test_model_capability_does_not_guess_deployment_names(self) -> None:
        from crewai.llms.base_llm import BaseLLM

        class TestLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None):
                return "test"

        llm = TestLLM(model="vision-production", provider="azure")
        assert llm.get_model_capability("supports_vision") is None

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
            "text": TextFile(source=b"hello"),
        }

        result = format_multimodal_content(files, getattr(llm, "provider", None) or llm.model)

        assert len(result) == 1

    def test_format_empty_files_dict(self) -> None:
        """Test empty files dict returns empty list."""
        llm = LLM(model="gpt-4o")

        result = format_multimodal_content({}, llm.model)

        assert result == []
