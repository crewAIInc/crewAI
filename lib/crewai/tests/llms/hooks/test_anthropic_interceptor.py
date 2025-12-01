"""Tests for Anthropic provider with interceptor integration."""

import os

import httpx
import pytest

from crewai.llm import LLM
from crewai.llms.hooks.base import BaseInterceptor


@pytest.fixture(autouse=True)
def setup_anthropic_api_key(monkeypatch):
    """Set dummy Anthropic API key for tests that don't make real API calls."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-dummy")


class AnthropicTestInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Test interceptor for Anthropic provider."""

    def __init__(self) -> None:
        """Initialize tracking and modification state."""
        self.outbound_calls: list[httpx.Request] = []
        self.inbound_calls: list[httpx.Response] = []
        self.custom_header_value = "anthropic-test-value"

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Track and modify outbound Anthropic requests.

        Args:
            message: The outbound request.

        Returns:
            Modified request with custom headers.
        """
        self.outbound_calls.append(message)
        message.headers["X-Anthropic-Interceptor"] = self.custom_header_value
        message.headers["X-Request-ID"] = "test-request-456"
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Track inbound Anthropic responses.

        Args:
            message: The inbound response.

        Returns:
            The response with tracking header.
        """
        self.inbound_calls.append(message)
        message.headers["X-Response-Tracked"] = "true"
        return message


class TestAnthropicInterceptorIntegration:
    """Test suite for Anthropic provider with interceptor."""

    def test_anthropic_llm_accepts_interceptor(self) -> None:
        """Test that Anthropic LLM accepts interceptor parameter."""
        interceptor = AnthropicTestInterceptor()

        llm = LLM(model="anthropic/claude-3-5-sonnet-20241022", interceptor=interceptor)

        assert llm.interceptor is interceptor

    @pytest.mark.vcr()
    def test_anthropic_call_with_interceptor_tracks_requests(self) -> None:
        """Test that interceptor tracks Anthropic API requests."""
        interceptor = AnthropicTestInterceptor()
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022", interceptor=interceptor)

        # Make a simple completion call
        result = llm.call(
            messages=[{"role": "user", "content": "Say 'Hello World' and nothing else"}]
        )

        # Verify custom headers were added
        for request in interceptor.outbound_calls:
            assert "X-Anthropic-Interceptor" in request.headers
            assert request.headers["X-Anthropic-Interceptor"] == "anthropic-test-value"
            assert "X-Request-ID" in request.headers
            assert request.headers["X-Request-ID"] == "test-request-456"

        # Verify response was tracked
        for response in interceptor.inbound_calls:
            assert "X-Response-Tracked" in response.headers
            assert response.headers["X-Response-Tracked"] == "true"

        # Verify result is valid
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_anthropic_without_interceptor_works(self) -> None:
        """Test that Anthropic LLM works without interceptor."""
        llm = LLM(model="anthropic/claude-3-5-sonnet-20241022")

        assert llm.interceptor is None

    def test_multiple_anthropic_llms_different_interceptors(self) -> None:
        """Test that multiple Anthropic LLMs can have different interceptors."""
        interceptor1 = AnthropicTestInterceptor()
        interceptor1.custom_header_value = "claude-opus-value"

        interceptor2 = AnthropicTestInterceptor()
        interceptor2.custom_header_value = "claude-sonnet-value"

        llm1 = LLM(model="anthropic/claude-3-opus-20240229", interceptor=interceptor1)
        llm2 = LLM(model="anthropic/claude-3-5-sonnet-20241022", interceptor=interceptor2)

        assert llm1.interceptor is interceptor1
        assert llm2.interceptor is interceptor2
        assert llm1.interceptor.custom_header_value == "claude-opus-value"
        assert llm2.interceptor.custom_header_value == "claude-sonnet-value"


class AnthropicLoggingInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Interceptor that logs Anthropic request/response details."""

    def __init__(self) -> None:
        """Initialize logging lists."""
        self.request_urls: list[str] = []
        self.request_methods: list[str] = []
        self.response_status_codes: list[int] = []
        self.anthropic_version_headers: list[str] = []

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Log outbound request details.

        Args:
            message: The outbound request.

        Returns:
            The request unchanged.
        """
        self.request_urls.append(str(message.url))
        self.request_methods.append(message.method)
        if "anthropic-version" in message.headers:
            self.anthropic_version_headers.append(message.headers["anthropic-version"])
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Log inbound response details.

        Args:
            message: The inbound response.

        Returns:
            The response unchanged.
        """
        self.response_status_codes.append(message.status_code)
        return message


class TestAnthropicLoggingInterceptor:
    """Test suite for logging interceptor with Anthropic."""

    def test_logging_interceptor_instantiation(self) -> None:
        """Test that logging interceptor can be created with Anthropic LLM."""
        interceptor = AnthropicLoggingInterceptor()
        llm = LLM(model="anthropic/claude-3-5-sonnet-20241022", interceptor=interceptor)

        assert llm.interceptor is interceptor
        assert isinstance(llm.interceptor, AnthropicLoggingInterceptor)

    @pytest.mark.vcr()
    def test_logging_interceptor_tracks_details(self) -> None:
        """Test that logging interceptor tracks request/response details."""
        interceptor = AnthropicLoggingInterceptor()
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022", interceptor=interceptor)

        # Make a completion call
        result = llm.call(messages=[{"role": "user", "content": "Count from 1 to 3"}])

        # Verify URL points to Anthropic API
        for url in interceptor.request_urls:
            assert "anthropic" in url.lower() or "api" in url.lower()

        # Verify methods are POST (messages endpoint uses POST)
        for method in interceptor.request_methods:
            assert method == "POST"

        # Verify successful status codes
        for status_code in interceptor.response_status_codes:
            assert 200 <= status_code < 300


        # Verify result is valid
        assert result is not None


class AnthropicHeaderInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Interceptor that adds Anthropic-specific headers."""

    def __init__(self, workspace_id: str, user_id: str) -> None:
        """Initialize with Anthropic-specific metadata.

        Args:
            workspace_id: The workspace ID to inject.
            user_id: The user ID to inject.
        """
        self.workspace_id = workspace_id
        self.user_id = user_id

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Add custom metadata headers to request.

        Args:
            message: The outbound request.

        Returns:
            Request with metadata headers.
        """
        message.headers["X-Workspace-ID"] = self.workspace_id
        message.headers["X-User-ID"] = self.user_id
        message.headers["X-Custom-Client"] = "crewai-interceptor"
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Pass through inbound response.

        Args:
            message: The inbound response.

        Returns:
            The response unchanged.
        """
        return message


class TestAnthropicHeaderInterceptor:
    """Test suite for header interceptor with Anthropic."""

    def test_header_interceptor_with_anthropic(self) -> None:
        """Test that header interceptor can be used with Anthropic LLM."""
        interceptor = AnthropicHeaderInterceptor(
            workspace_id="ws-789", user_id="user-012"
        )
        llm = LLM(model="anthropic/claude-3-5-sonnet-20241022", interceptor=interceptor)

        assert llm.interceptor is interceptor
        assert llm.interceptor.workspace_id == "ws-789"
        assert llm.interceptor.user_id == "user-012"

    def test_header_interceptor_adds_headers(self) -> None:
        """Test that header interceptor adds custom headers to requests."""
        interceptor = AnthropicHeaderInterceptor(workspace_id="ws-123", user_id="u-456")
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

        modified_request = interceptor.on_outbound(request)

        assert "X-Workspace-ID" in modified_request.headers
        assert modified_request.headers["X-Workspace-ID"] == "ws-123"
        assert "X-User-ID" in modified_request.headers
        assert modified_request.headers["X-User-ID"] == "u-456"
        assert "X-Custom-Client" in modified_request.headers
        assert modified_request.headers["X-Custom-Client"] == "crewai-interceptor"

    @pytest.mark.vcr()
    def test_header_interceptor_with_real_call(self) -> None:
        """Test that header interceptor works with real Anthropic API call."""
        interceptor = AnthropicHeaderInterceptor(workspace_id="ws-999", user_id="u-888")
        llm = LLM(model="anthropic/claude-3-5-haiku-20241022", interceptor=interceptor)

        # Make a simple call
        result = llm.call(
            messages=[{"role": "user", "content": "Reply with just the word: SUCCESS"}]
        )

        # Verify the call succeeded
        assert result is not None
        assert len(result) > 0

        # Verify the interceptor was configured
        assert llm.interceptor is interceptor


class TestMixedProviderInterceptors:
    """Test suite for using interceptors with different providers."""

    def test_openai_and_anthropic_different_interceptors(self) -> None:
        """Test that OpenAI and Anthropic LLMs can have different interceptors."""
        openai_interceptor = AnthropicTestInterceptor()
        openai_interceptor.custom_header_value = "openai-specific"

        anthropic_interceptor = AnthropicTestInterceptor()
        anthropic_interceptor.custom_header_value = "anthropic-specific"

        openai_llm = LLM(model="gpt-4", interceptor=openai_interceptor)
        anthropic_llm = LLM(
            model="anthropic/claude-3-5-sonnet-20241022", interceptor=anthropic_interceptor
        )

        assert openai_llm.interceptor is openai_interceptor
        assert anthropic_llm.interceptor is anthropic_interceptor
        assert openai_llm.interceptor.custom_header_value == "openai-specific"
        assert anthropic_llm.interceptor.custom_header_value == "anthropic-specific"

    def test_same_interceptor_different_providers(self) -> None:
        """Test that same interceptor instance can be used with multiple providers."""
        shared_interceptor = AnthropicTestInterceptor()

        openai_llm = LLM(model="gpt-4", interceptor=shared_interceptor)
        anthropic_llm = LLM(
            model="anthropic/claude-3-5-sonnet-20241022", interceptor=shared_interceptor
        )

        assert openai_llm.interceptor is shared_interceptor
        assert anthropic_llm.interceptor is shared_interceptor
        assert openai_llm.interceptor is anthropic_llm.interceptor
