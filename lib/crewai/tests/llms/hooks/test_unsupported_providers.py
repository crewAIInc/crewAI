"""Tests for interceptor behavior with unsupported providers."""

import os

import httpx
import pytest

from crewai.llm import LLM
from crewai.llms.hooks.base import BaseInterceptor


@pytest.fixture(autouse=True)
def setup_provider_api_keys(monkeypatch):
    """Set dummy API keys for providers that require them."""
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-dummy")
    if "ANTHROPIC_API_KEY" not in os.environ:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-dummy")
    if "GOOGLE_API_KEY" not in os.environ:
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-dummy")


class DummyInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Simple dummy interceptor for testing."""

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Pass through outbound request.

        Args:
            message: The outbound request.

        Returns:
            The request unchanged.
        """
        message.headers["X-Dummy"] = "true"
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Pass through inbound response.

        Args:
            message: The inbound response.

        Returns:
            The response unchanged.
        """
        return message


class TestAzureProviderInterceptor:
    """Test suite for Azure provider with interceptor (unsupported)."""

    def test_azure_llm_accepts_interceptor_parameter(self) -> None:
        """Test that Azure LLM raises NotImplementedError with interceptor."""
        interceptor = DummyInterceptor()

        # Azure provider should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="azure/gpt-4",
                interceptor=interceptor,
                api_key="test-key",
                endpoint="https://test.openai.azure.com/openai/deployments/gpt-4",
            )

        assert "interceptor" in str(exc_info.value).lower()

    def test_azure_raises_not_implemented_on_initialization(self) -> None:
        """Test that Azure raises NotImplementedError when interceptor is used."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="azure/gpt-4",
                interceptor=interceptor,
                api_key="test-key",
                endpoint="https://test.openai.azure.com/openai/deployments/gpt-4",
            )

        error_msg = str(exc_info.value).lower()
        assert "interceptor" in error_msg
        assert "azure" in error_msg

    def test_azure_without_interceptor_works(self) -> None:
        """Test that Azure LLM works without interceptor."""
        llm = LLM(
            model="azure/gpt-4",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/openai/deployments/gpt-4",
        )

        # Azure provider doesn't have interceptor attribute
        assert not hasattr(llm, 'interceptor') or llm.interceptor is None


class TestBedrockProviderInterceptor:
    """Test suite for Bedrock provider with interceptor (unsupported)."""

    def test_bedrock_llm_accepts_interceptor_parameter(self) -> None:
        """Test that Bedrock LLM raises NotImplementedError with interceptor."""
        interceptor = DummyInterceptor()

        # Bedrock provider should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                interceptor=interceptor,
                aws_access_key_id="test-access-key",
                aws_secret_access_key="test-secret-key",
                aws_region_name="us-east-1",
            )

        error_msg = str(exc_info.value).lower()
        assert "interceptor" in error_msg
        assert "bedrock" in error_msg

    def test_bedrock_raises_not_implemented_on_initialization(self) -> None:
        """Test that Bedrock raises NotImplementedError when interceptor is used."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                interceptor=interceptor,
                aws_access_key_id="test-access-key",
                aws_secret_access_key="test-secret-key",
                aws_region_name="us-east-1",
            )

        error_msg = str(exc_info.value).lower()
        assert "interceptor" in error_msg
        assert "bedrock" in error_msg

    def test_bedrock_without_interceptor_works(self) -> None:
        """Test that Bedrock LLM works without interceptor."""
        llm = LLM(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_region_name="us-east-1",
        )

        # Bedrock provider doesn't have interceptor attribute
        assert not hasattr(llm, 'interceptor') or llm.interceptor is None


class TestGeminiProviderInterceptor:
    """Test suite for Gemini provider with interceptor (unsupported)."""

    def test_gemini_llm_accepts_interceptor_parameter(self) -> None:
        """Test that Gemini LLM raises NotImplementedError with interceptor."""
        interceptor = DummyInterceptor()

        # Gemini provider should raise NotImplementedError
        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="gemini/gemini-2.5-pro",
                interceptor=interceptor,
                api_key="test-gemini-key",
            )

        error_msg = str(exc_info.value).lower()
        assert "interceptor" in error_msg
        assert "gemini" in error_msg

    def test_gemini_raises_not_implemented_on_initialization(self) -> None:
        """Test that Gemini raises NotImplementedError when interceptor is used."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="gemini/gemini-2.5-pro",
                interceptor=interceptor,
                api_key="test-gemini-key",
            )

        error_msg = str(exc_info.value).lower()
        assert "interceptor" in error_msg
        assert "gemini" in error_msg

    def test_gemini_without_interceptor_works(self) -> None:
        """Test that Gemini LLM works without interceptor."""
        llm = LLM(
            model="gemini/gemini-2.5-pro",
            api_key="test-gemini-key",
        )

        # Gemini provider doesn't have interceptor attribute
        assert not hasattr(llm, 'interceptor') or llm.interceptor is None


class TestUnsupportedProviderMessages:
    """Test suite for error messages from unsupported providers."""

    def test_azure_error_message_is_clear(self) -> None:
        """Test that Azure error message clearly states lack of support."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="azure/gpt-4",
                interceptor=interceptor,
                api_key="test-key",
                endpoint="https://test.openai.azure.com/openai/deployments/gpt-4",
            )

        error_message = str(exc_info.value).lower()
        assert "azure" in error_message
        assert "interceptor" in error_message

    def test_bedrock_error_message_is_clear(self) -> None:
        """Test that Bedrock error message clearly states lack of support."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                interceptor=interceptor,
                aws_access_key_id="test-access-key",
                aws_secret_access_key="test-secret-key",
                aws_region_name="us-east-1",
            )

        error_message = str(exc_info.value).lower()
        assert "bedrock" in error_message
        assert "interceptor" in error_message

    def test_gemini_error_message_is_clear(self) -> None:
        """Test that Gemini error message clearly states lack of support."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="gemini/gemini-2.5-pro",
                interceptor=interceptor,
                api_key="test-gemini-key",
            )

        error_message = str(exc_info.value).lower()
        assert "gemini" in error_message
        assert "interceptor" in error_message


class TestProviderSupportMatrix:
    """Test suite to document which providers support interceptors."""

    def test_supported_providers_accept_interceptor(self) -> None:
        """Test that supported providers accept and use interceptors."""
        interceptor = DummyInterceptor()

        # OpenAI - SUPPORTED
        openai_llm = LLM(model="gpt-4", interceptor=interceptor)
        assert openai_llm.interceptor is interceptor

        # Anthropic - SUPPORTED
        anthropic_llm = LLM(model="anthropic/claude-3-opus-20240229", interceptor=interceptor)
        assert anthropic_llm.interceptor is interceptor

    def test_unsupported_providers_raise_error(self) -> None:
        """Test that unsupported providers raise NotImplementedError."""
        interceptor = DummyInterceptor()

        # Azure - NOT SUPPORTED
        with pytest.raises(NotImplementedError):
            LLM(
                model="azure/gpt-4",
                interceptor=interceptor,
                api_key="test",
                endpoint="https://test.openai.azure.com/openai/deployments/gpt-4",
            )

        # Bedrock - NOT SUPPORTED
        with pytest.raises(NotImplementedError):
            LLM(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                interceptor=interceptor,
                aws_access_key_id="test",
                aws_secret_access_key="test",
                aws_region_name="us-east-1",
            )

        # Gemini - NOT SUPPORTED
        with pytest.raises(NotImplementedError):
            LLM(
                model="gemini/gemini-2.5-pro",
                interceptor=interceptor,
                api_key="test",
            )

    def test_all_providers_work_without_interceptor(self) -> None:
        """Test that all providers work normally without interceptor."""
        # OpenAI
        openai_llm = LLM(model="gpt-4")
        assert openai_llm.interceptor is None

        # Anthropic
        anthropic_llm = LLM(model="anthropic/claude-3-opus-20240229")
        assert anthropic_llm.interceptor is None

        # Azure - doesn't have interceptor attribute
        azure_llm = LLM(
            model="azure/gpt-4",
            api_key="test",
            endpoint="https://test.openai.azure.com/openai/deployments/gpt-4",
        )
        assert not hasattr(azure_llm, 'interceptor') or azure_llm.interceptor is None

        # Bedrock - doesn't have interceptor attribute
        bedrock_llm = LLM(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_region_name="us-east-1",
        )
        assert not hasattr(bedrock_llm, 'interceptor') or bedrock_llm.interceptor is None

        # Gemini - doesn't have interceptor attribute
        gemini_llm = LLM(model="gemini/gemini-2.5-pro", api_key="test")
        assert not hasattr(gemini_llm, 'interceptor') or gemini_llm.interceptor is None
