"""Tests for interceptor behavior with unsupported providers."""

import httpx
import pytest

from crewai.llm import LLM
from crewai.llms.hooks.base import BaseInterceptor


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
        """Test that Azure LLM accepts interceptor parameter without error."""
        interceptor = DummyInterceptor()

        # Azure provider should accept the parameter
        llm = LLM(
            model="azure/gpt-4",
            interceptor=interceptor,
            api_key="test-key",
            api_base="https://test.openai.azure.com",
        )

        assert llm.interceptor is interceptor

    def test_azure_raises_not_implemented_on_initialization(self) -> None:
        """Test that Azure raises NotImplementedError when interceptor is used."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="azure/gpt-4",
                interceptor=interceptor,
                api_key="test-key",
                api_base="https://test.openai.azure.com",
            )

        assert "Azure provider does not yet support interceptors" in str(exc_info.value)

    def test_azure_without_interceptor_works(self) -> None:
        """Test that Azure LLM works without interceptor."""
        llm = LLM(
            model="azure/gpt-4",
            api_key="test-key",
            api_base="https://test.openai.azure.com",
        )

        assert llm.interceptor is None


class TestBedrockProviderInterceptor:
    """Test suite for Bedrock provider with interceptor (unsupported)."""

    def test_bedrock_llm_accepts_interceptor_parameter(self) -> None:
        """Test that Bedrock LLM accepts interceptor parameter without error."""
        interceptor = DummyInterceptor()

        # Bedrock provider should accept the parameter
        llm = LLM(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            interceptor=interceptor,
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_region_name="us-east-1",
        )

        assert llm.interceptor is interceptor

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

        assert "Bedrock provider does not yet support interceptors" in str(exc_info.value)

    def test_bedrock_without_interceptor_works(self) -> None:
        """Test that Bedrock LLM works without interceptor."""
        llm = LLM(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            aws_access_key_id="test-access-key",
            aws_secret_access_key="test-secret-key",
            aws_region_name="us-east-1",
        )

        assert llm.interceptor is None


class TestGeminiProviderInterceptor:
    """Test suite for Gemini provider with interceptor (unsupported)."""

    def test_gemini_llm_accepts_interceptor_parameter(self) -> None:
        """Test that Gemini LLM accepts interceptor parameter without error."""
        interceptor = DummyInterceptor()

        # Gemini provider should accept the parameter
        llm = LLM(
            model="gemini/gemini-pro",
            interceptor=interceptor,
            api_key="test-gemini-key",
        )

        assert llm.interceptor is interceptor

    def test_gemini_raises_not_implemented_on_initialization(self) -> None:
        """Test that Gemini raises NotImplementedError when interceptor is used."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="gemini/gemini-pro",
                interceptor=interceptor,
                api_key="test-gemini-key",
            )

        assert "Gemini provider does not yet support interceptors" in str(exc_info.value)

    def test_gemini_without_interceptor_works(self) -> None:
        """Test that Gemini LLM works without interceptor."""
        llm = LLM(
            model="gemini/gemini-pro",
            api_key="test-gemini-key",
        )

        assert llm.interceptor is None


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
                api_base="https://test.openai.azure.com",
            )

        error_message = str(exc_info.value)
        assert "Azure" in error_message
        assert "does not yet support interceptors" in error_message

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

        error_message = str(exc_info.value)
        assert "Bedrock" in error_message
        assert "does not yet support interceptors" in error_message

    def test_gemini_error_message_is_clear(self) -> None:
        """Test that Gemini error message clearly states lack of support."""
        interceptor = DummyInterceptor()

        with pytest.raises(NotImplementedError) as exc_info:
            LLM(
                model="gemini/gemini-pro",
                interceptor=interceptor,
                api_key="test-gemini-key",
            )

        error_message = str(exc_info.value)
        assert "Gemini" in error_message
        assert "does not yet support interceptors" in error_message


class TestProviderSupportMatrix:
    """Test suite to document which providers support interceptors."""

    def test_supported_providers_accept_interceptor(self) -> None:
        """Test that supported providers accept and use interceptors."""
        interceptor = DummyInterceptor()

        # OpenAI - SUPPORTED
        openai_llm = LLM(model="gpt-4", interceptor=interceptor)
        assert openai_llm.interceptor is interceptor

        # Anthropic - SUPPORTED
        anthropic_llm = LLM(model="claude-3-opus-20240229", interceptor=interceptor)
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
                api_base="https://test.openai.azure.com",
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
                model="gemini/gemini-pro",
                interceptor=interceptor,
                api_key="test",
            )

    def test_all_providers_work_without_interceptor(self) -> None:
        """Test that all providers work normally without interceptor."""
        # OpenAI
        openai_llm = LLM(model="gpt-4")
        assert openai_llm.interceptor is None

        # Anthropic
        anthropic_llm = LLM(model="claude-3-opus-20240229")
        assert anthropic_llm.interceptor is None

        # Azure
        azure_llm = LLM(
            model="azure/gpt-4",
            api_key="test",
            api_base="https://test.openai.azure.com",
        )
        assert azure_llm.interceptor is None

        # Bedrock
        bedrock_llm = LLM(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_region_name="us-east-1",
        )
        assert bedrock_llm.interceptor is None

        # Gemini
        gemini_llm = LLM(model="gemini/gemini-pro", api_key="test")
        assert gemini_llm.interceptor is None