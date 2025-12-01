"""Tests for OpenAI provider with interceptor integration."""

import httpx
import pytest

from crewai.llm import LLM
from crewai.llms.hooks.base import BaseInterceptor


class OpenAITestInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Test interceptor for OpenAI provider."""

    def __init__(self) -> None:
        """Initialize tracking and modification state."""
        self.outbound_calls: list[httpx.Request] = []
        self.inbound_calls: list[httpx.Response] = []
        self.custom_header_value = "openai-test-value"

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Track and modify outbound OpenAI requests.

        Args:
            message: The outbound request.

        Returns:
            Modified request with custom headers.
        """
        self.outbound_calls.append(message)
        message.headers["X-OpenAI-Interceptor"] = self.custom_header_value
        message.headers["X-Request-ID"] = "test-request-123"
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Track inbound OpenAI responses.

        Args:
            message: The inbound response.

        Returns:
            The response with tracking header.
        """
        self.inbound_calls.append(message)
        message.headers["X-Response-Tracked"] = "true"
        return message


class TestOpenAIInterceptorIntegration:
    """Test suite for OpenAI provider with interceptor."""

    def test_openai_llm_accepts_interceptor(self) -> None:
        """Test that OpenAI LLM accepts interceptor parameter."""
        interceptor = OpenAITestInterceptor()

        llm = LLM(model="gpt-4", interceptor=interceptor)

        assert llm.interceptor is interceptor

    @pytest.mark.vcr()
    def test_openai_call_with_interceptor_tracks_requests(self) -> None:
        """Test that interceptor tracks OpenAI API requests."""
        interceptor = OpenAITestInterceptor()
        llm = LLM(model="gpt-4o-mini", interceptor=interceptor)

        # Make a simple completion call
        result = llm.call(
            messages=[{"role": "user", "content": "Say 'Hello World' and nothing else"}]
        )

        # Verify custom headers were added
        for request in interceptor.outbound_calls:
            assert "X-OpenAI-Interceptor" in request.headers
            assert request.headers["X-OpenAI-Interceptor"] == "openai-test-value"
            assert "X-Request-ID" in request.headers
            assert request.headers["X-Request-ID"] == "test-request-123"

        # Verify response was tracked
        for response in interceptor.inbound_calls:
            assert "X-Response-Tracked" in response.headers
            assert response.headers["X-Response-Tracked"] == "true"

        # Verify result is valid
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_openai_without_interceptor_works(self) -> None:
        """Test that OpenAI LLM works without interceptor."""
        llm = LLM(model="gpt-4")

        assert llm.interceptor is None

    def test_multiple_openai_llms_different_interceptors(self) -> None:
        """Test that multiple OpenAI LLMs can have different interceptors."""
        interceptor1 = OpenAITestInterceptor()
        interceptor1.custom_header_value = "llm1-value"

        interceptor2 = OpenAITestInterceptor()
        interceptor2.custom_header_value = "llm2-value"

        llm1 = LLM(model="gpt-4", interceptor=interceptor1)
        llm2 = LLM(model="gpt-3.5-turbo", interceptor=interceptor2)

        assert llm1.interceptor is interceptor1
        assert llm2.interceptor is interceptor2
        assert llm1.interceptor.custom_header_value == "llm1-value"
        assert llm2.interceptor.custom_header_value == "llm2-value"


class LoggingInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Interceptor that logs request/response details for testing."""

    def __init__(self) -> None:
        """Initialize logging lists."""
        self.request_urls: list[str] = []
        self.request_methods: list[str] = []
        self.response_status_codes: list[int] = []

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Log outbound request details.

        Args:
            message: The outbound request.

        Returns:
            The request unchanged.
        """
        self.request_urls.append(str(message.url))
        self.request_methods.append(message.method)
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


class TestOpenAILoggingInterceptor:
    """Test suite for logging interceptor with OpenAI."""

    def test_logging_interceptor_instantiation(self) -> None:
        """Test that logging interceptor can be created with OpenAI LLM."""
        interceptor = LoggingInterceptor()
        llm = LLM(model="gpt-4", interceptor=interceptor)

        assert llm.interceptor is interceptor
        assert isinstance(llm.interceptor, LoggingInterceptor)

    @pytest.mark.vcr()
    def test_logging_interceptor_tracks_details(self) -> None:
        """Test that logging interceptor tracks request/response details."""
        interceptor = LoggingInterceptor()
        llm = LLM(model="gpt-4o-mini", interceptor=interceptor)

        # Make a completion call
        result = llm.call(
            messages=[{"role": "user", "content": "Count from 1 to 3"}]
        )

        # Verify URL points to OpenAI API
        for url in interceptor.request_urls:
            assert "openai" in url.lower() or "api" in url.lower()

        # Verify methods are POST (chat completions use POST)
        for method in interceptor.request_methods:
            assert method == "POST"

        # Verify successful status codes
        for status_code in interceptor.response_status_codes:
            assert 200 <= status_code < 300

        # Verify result is valid
        assert result is not None


class AuthInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Interceptor that adds authentication headers."""

    def __init__(self, api_key: str, org_id: str) -> None:
        """Initialize with auth credentials.

        Args:
            api_key: The API key to inject.
            org_id: The organization ID to inject.
        """
        self.api_key = api_key
        self.org_id = org_id

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Add authentication headers to request.

        Args:
            message: The outbound request.

        Returns:
            Request with auth headers.
        """
        message.headers["X-Custom-API-Key"] = self.api_key
        message.headers["X-Organization-ID"] = self.org_id
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Pass through inbound response.

        Args:
            message: The inbound response.

        Returns:
            The response unchanged.
        """
        return message


class TestOpenAIAuthInterceptor:
    """Test suite for authentication interceptor with OpenAI."""

    def test_auth_interceptor_with_openai(self) -> None:
        """Test that auth interceptor can be used with OpenAI LLM."""
        interceptor = AuthInterceptor(api_key="custom-key-123", org_id="org-456")
        llm = LLM(model="gpt-4", interceptor=interceptor)

        assert llm.interceptor is interceptor
        assert llm.interceptor.api_key == "custom-key-123"
        assert llm.interceptor.org_id == "org-456"

    def test_auth_interceptor_adds_headers(self) -> None:
        """Test that auth interceptor adds custom headers to requests."""
        interceptor = AuthInterceptor(api_key="test-key", org_id="test-org")
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

        modified_request = interceptor.on_outbound(request)

        assert "X-Custom-API-Key" in modified_request.headers
        assert modified_request.headers["X-Custom-API-Key"] == "test-key"
        assert "X-Organization-ID" in modified_request.headers
        assert modified_request.headers["X-Organization-ID"] == "test-org"

    @pytest.mark.vcr()
    def test_auth_interceptor_with_real_call(self) -> None:
        """Test that auth interceptor works with real OpenAI API call."""
        interceptor = AuthInterceptor(api_key="custom-123", org_id="org-789")
        llm = LLM(model="gpt-4o-mini", interceptor=interceptor)

        # Make a simple call
        result = llm.call(
            messages=[{"role": "user", "content": "Reply with just the word: SUCCESS"}]
        )

        # Verify the call succeeded
        assert result is not None
        assert len(result) > 0

        # Verify headers were added to outbound requests
        # (We can't directly inspect the request sent to OpenAI in this test,
        # but we verify the interceptor was configured and the call succeeded)
        assert llm.interceptor is interceptor
