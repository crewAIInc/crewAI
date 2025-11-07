"""Tests for base interceptor functionality."""

import httpx
import pytest

from crewai.llms.hooks.base import BaseInterceptor


class SimpleInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Simple test interceptor implementation."""

    def __init__(self) -> None:
        """Initialize tracking lists."""
        self.outbound_calls: list[httpx.Request] = []
        self.inbound_calls: list[httpx.Response] = []

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Track outbound calls.

        Args:
            message: The outbound request.

        Returns:
            The request unchanged.
        """
        self.outbound_calls.append(message)
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Track inbound calls.

        Args:
            message: The inbound response.

        Returns:
            The response unchanged.
        """
        self.inbound_calls.append(message)
        return message


class ModifyingInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Interceptor that modifies requests and responses."""

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Add custom header to outbound request.

        Args:
            message: The outbound request.

        Returns:
            Modified request with custom header.
        """
        message.headers["X-Custom-Header"] = "test-value"
        message.headers["X-Intercepted"] = "true"
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Add custom header to inbound response.

        Args:
            message: The inbound response.

        Returns:
            Modified response with custom header.
        """
        message.headers["X-Response-Intercepted"] = "true"
        return message


class AsyncInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Interceptor with async support."""

    def __init__(self) -> None:
        """Initialize tracking lists."""
        self.async_outbound_calls: list[httpx.Request] = []
        self.async_inbound_calls: list[httpx.Response] = []

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Handle sync outbound.

        Args:
            message: The outbound request.

        Returns:
            The request unchanged.
        """
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Handle sync inbound.

        Args:
            message: The inbound response.

        Returns:
            The response unchanged.
        """
        return message

    async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
        """Handle async outbound.

        Args:
            message: The outbound request.

        Returns:
            Modified request with async header.
        """
        self.async_outbound_calls.append(message)
        message.headers["X-Async-Outbound"] = "true"
        return message

    async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
        """Handle async inbound.

        Args:
            message: The inbound response.

        Returns:
            Modified response with async header.
        """
        self.async_inbound_calls.append(message)
        message.headers["X-Async-Inbound"] = "true"
        return message


class TestBaseInterceptor:
    """Test suite for BaseInterceptor class."""

    def test_interceptor_instantiation(self) -> None:
        """Test that interceptor can be instantiated."""
        interceptor = SimpleInterceptor()
        assert interceptor is not None
        assert isinstance(interceptor, BaseInterceptor)

    def test_on_outbound_called(self) -> None:
        """Test that on_outbound is called and tracks requests."""
        interceptor = SimpleInterceptor()
        request = httpx.Request("GET", "https://api.example.com/test")

        result = interceptor.on_outbound(request)

        assert len(interceptor.outbound_calls) == 1
        assert interceptor.outbound_calls[0] is request
        assert result is request

    def test_on_inbound_called(self) -> None:
        """Test that on_inbound is called and tracks responses."""
        interceptor = SimpleInterceptor()
        response = httpx.Response(200, json={"status": "ok"})

        result = interceptor.on_inbound(response)

        assert len(interceptor.inbound_calls) == 1
        assert interceptor.inbound_calls[0] is response
        assert result is response

    def test_multiple_outbound_calls(self) -> None:
        """Test that interceptor tracks multiple outbound calls."""
        interceptor = SimpleInterceptor()
        requests = [
            httpx.Request("GET", "https://api.example.com/1"),
            httpx.Request("POST", "https://api.example.com/2"),
            httpx.Request("PUT", "https://api.example.com/3"),
        ]

        for req in requests:
            interceptor.on_outbound(req)

        assert len(interceptor.outbound_calls) == 3
        assert interceptor.outbound_calls == requests

    def test_multiple_inbound_calls(self) -> None:
        """Test that interceptor tracks multiple inbound calls."""
        interceptor = SimpleInterceptor()
        responses = [
            httpx.Response(200, json={"id": 1}),
            httpx.Response(201, json={"id": 2}),
            httpx.Response(404, json={"error": "not found"}),
        ]

        for resp in responses:
            interceptor.on_inbound(resp)

        assert len(interceptor.inbound_calls) == 3
        assert interceptor.inbound_calls == responses


class TestModifyingInterceptor:
    """Test suite for interceptor that modifies messages."""

    def test_outbound_header_modification(self) -> None:
        """Test that interceptor can add headers to outbound requests."""
        interceptor = ModifyingInterceptor()
        request = httpx.Request("GET", "https://api.example.com/test")

        result = interceptor.on_outbound(request)

        assert result is request
        assert "X-Custom-Header" in result.headers
        assert result.headers["X-Custom-Header"] == "test-value"
        assert "X-Intercepted" in result.headers
        assert result.headers["X-Intercepted"] == "true"

    def test_inbound_header_modification(self) -> None:
        """Test that interceptor can add headers to inbound responses."""
        interceptor = ModifyingInterceptor()
        response = httpx.Response(200, json={"status": "ok"})

        result = interceptor.on_inbound(response)

        assert result is response
        assert "X-Response-Intercepted" in result.headers
        assert result.headers["X-Response-Intercepted"] == "true"

    def test_preserves_existing_headers(self) -> None:
        """Test that interceptor preserves existing headers."""
        interceptor = ModifyingInterceptor()
        request = httpx.Request(
            "GET",
            "https://api.example.com/test",
            headers={"Authorization": "Bearer token123", "Content-Type": "application/json"},
        )

        result = interceptor.on_outbound(request)

        assert result.headers["Authorization"] == "Bearer token123"
        assert result.headers["Content-Type"] == "application/json"
        assert result.headers["X-Custom-Header"] == "test-value"


class TestAsyncInterceptor:
    """Test suite for async interceptor functionality."""

    def test_sync_methods_work(self) -> None:
        """Test that sync methods still work on async interceptor."""
        interceptor = AsyncInterceptor()
        request = httpx.Request("GET", "https://api.example.com/test")
        response = httpx.Response(200)

        req_result = interceptor.on_outbound(request)
        resp_result = interceptor.on_inbound(response)

        assert req_result is request
        assert resp_result is response

    @pytest.mark.asyncio
    async def test_async_outbound(self) -> None:
        """Test async outbound hook."""
        interceptor = AsyncInterceptor()
        request = httpx.Request("GET", "https://api.example.com/test")

        result = await interceptor.aon_outbound(request)

        assert result is request
        assert len(interceptor.async_outbound_calls) == 1
        assert interceptor.async_outbound_calls[0] is request
        assert "X-Async-Outbound" in result.headers
        assert result.headers["X-Async-Outbound"] == "true"

    @pytest.mark.asyncio
    async def test_async_inbound(self) -> None:
        """Test async inbound hook."""
        interceptor = AsyncInterceptor()
        response = httpx.Response(200, json={"status": "ok"})

        result = await interceptor.aon_inbound(response)

        assert result is response
        assert len(interceptor.async_inbound_calls) == 1
        assert interceptor.async_inbound_calls[0] is response
        assert "X-Async-Inbound" in result.headers
        assert result.headers["X-Async-Inbound"] == "true"

    @pytest.mark.asyncio
    async def test_default_async_not_implemented(self) -> None:
        """Test that default async methods raise NotImplementedError."""
        interceptor = SimpleInterceptor()
        request = httpx.Request("GET", "https://api.example.com/test")
        response = httpx.Response(200)

        with pytest.raises(NotImplementedError):
            await interceptor.aon_outbound(request)

        with pytest.raises(NotImplementedError):
            await interceptor.aon_inbound(response)
