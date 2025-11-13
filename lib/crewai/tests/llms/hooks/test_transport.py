"""Tests for transport layer with interceptor integration."""

from unittest.mock import Mock

import httpx
import pytest

from crewai.llms.hooks.base import BaseInterceptor
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport


class TrackingInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """Test interceptor that tracks all calls."""

    def __init__(self) -> None:
        """Initialize tracking lists."""
        self.outbound_calls: list[httpx.Request] = []
        self.inbound_calls: list[httpx.Response] = []
        self.async_outbound_calls: list[httpx.Request] = []
        self.async_inbound_calls: list[httpx.Response] = []

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        """Track outbound calls and add header.

        Args:
            message: The outbound request.

        Returns:
            Modified request with tracking header.
        """
        self.outbound_calls.append(message)
        message.headers["X-Intercepted-Sync"] = "true"
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        """Track inbound calls.

        Args:
            message: The inbound response.

        Returns:
            The response with tracking header.
        """
        self.inbound_calls.append(message)
        message.headers["X-Response-Intercepted-Sync"] = "true"
        return message

    async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
        """Track async outbound calls and add header.

        Args:
            message: The outbound request.

        Returns:
            Modified request with tracking header.
        """
        self.async_outbound_calls.append(message)
        message.headers["X-Intercepted-Async"] = "true"
        return message

    async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
        """Track async inbound calls.

        Args:
            message: The inbound response.

        Returns:
            The response with tracking header.
        """
        self.async_inbound_calls.append(message)
        message.headers["X-Response-Intercepted-Async"] = "true"
        return message


class TestHTTPTransport:
    """Test suite for sync HTTPTransport with interceptor."""

    def test_transport_instantiation(self) -> None:
        """Test that transport can be instantiated with interceptor."""
        interceptor = TrackingInterceptor()
        transport = HTTPTransport(interceptor=interceptor)

        assert transport.interceptor is interceptor

    def test_transport_requires_interceptor(self) -> None:
        """Test that transport requires interceptor parameter."""
        # HTTPTransport requires an interceptor parameter
        with pytest.raises(TypeError):
            HTTPTransport()

    def test_interceptor_called_on_request(self) -> None:
        """Test that interceptor hooks are called during request handling."""
        interceptor = TrackingInterceptor()
        transport = HTTPTransport(interceptor=interceptor)

        # Create a mock parent transport that returns a response
        mock_response = httpx.Response(200, json={"success": True})
        mock_parent_handle = Mock(return_value=mock_response)

        # Monkey-patch the parent's handle_request
        original_handle = httpx.HTTPTransport.handle_request
        httpx.HTTPTransport.handle_request = mock_parent_handle

        try:
            request = httpx.Request("GET", "https://api.example.com/test")
            response = transport.handle_request(request)

            # Verify interceptor was called
            assert len(interceptor.outbound_calls) == 1
            assert len(interceptor.inbound_calls) == 1
            assert interceptor.outbound_calls[0] is request
            assert interceptor.inbound_calls[0] is response

            # Verify headers were added
            assert "X-Intercepted-Sync" in request.headers
            assert request.headers["X-Intercepted-Sync"] == "true"
            assert "X-Response-Intercepted-Sync" in response.headers
            assert response.headers["X-Response-Intercepted-Sync"] == "true"
        finally:
            # Restore original method
            httpx.HTTPTransport.handle_request = original_handle



class TestAsyncHTTPTransport:
    """Test suite for async AsyncHTTPransport with interceptor."""

    def test_async_transport_instantiation(self) -> None:
        """Test that async transport can be instantiated with interceptor."""
        interceptor = TrackingInterceptor()
        transport = AsyncHTTPTransport(interceptor=interceptor)

        assert transport.interceptor is interceptor

    def test_async_transport_requires_interceptor(self) -> None:
        """Test that async transport requires interceptor parameter."""
        # AsyncHTTPransport requires an interceptor parameter
        with pytest.raises(TypeError):
            AsyncHTTPTransport()

    @pytest.mark.asyncio
    async def test_async_interceptor_called_on_request(self) -> None:
        """Test that async interceptor hooks are called during request handling."""
        interceptor = TrackingInterceptor()
        transport = AsyncHTTPTransport(interceptor=interceptor)

        # Create a mock parent transport that returns a response
        mock_response = httpx.Response(200, json={"success": True})

        async def mock_handle(*args, **kwargs):
            return mock_response

        mock_parent_handle = Mock(side_effect=mock_handle)

        # Monkey-patch the parent's handle_async_request
        original_handle = httpx.AsyncHTTPTransport.handle_async_request
        httpx.AsyncHTTPTransport.handle_async_request = mock_parent_handle

        try:
            request = httpx.Request("GET", "https://api.example.com/test")
            response = await transport.handle_async_request(request)

            # Verify async interceptor was called
            assert len(interceptor.async_outbound_calls) == 1
            assert len(interceptor.async_inbound_calls) == 1
            assert interceptor.async_outbound_calls[0] is request
            assert interceptor.async_inbound_calls[0] is response

            # Verify sync interceptor was NOT called
            assert len(interceptor.outbound_calls) == 0
            assert len(interceptor.inbound_calls) == 0

            # Verify async headers were added
            assert "X-Intercepted-Async" in request.headers
            assert request.headers["X-Intercepted-Async"] == "true"
            assert "X-Response-Intercepted-Async" in response.headers
            assert response.headers["X-Response-Intercepted-Async"] == "true"
        finally:
            # Restore original method
            httpx.AsyncHTTPTransport.handle_async_request = original_handle



class TestTransportIntegration:
    """Test suite for transport integration scenarios."""

    def test_multiple_requests_same_interceptor(self) -> None:
        """Test that multiple requests through same interceptor are tracked."""
        interceptor = TrackingInterceptor()
        transport = HTTPTransport(interceptor=interceptor)

        mock_response = httpx.Response(200)
        mock_parent_handle = Mock(return_value=mock_response)

        original_handle = httpx.HTTPTransport.handle_request
        httpx.HTTPTransport.handle_request = mock_parent_handle

        try:
            # Make multiple requests
            requests = [
                httpx.Request("GET", "https://api.example.com/1"),
                httpx.Request("POST", "https://api.example.com/2"),
                httpx.Request("PUT", "https://api.example.com/3"),
            ]

            for req in requests:
                transport.handle_request(req)

            # Verify all requests were intercepted
            assert len(interceptor.outbound_calls) == 3
            assert len(interceptor.inbound_calls) == 3
            assert interceptor.outbound_calls == requests
        finally:
            httpx.HTTPTransport.handle_request = original_handle

    @pytest.mark.asyncio
    async def test_multiple_async_requests_same_interceptor(self) -> None:
        """Test that multiple async requests through same interceptor are tracked."""
        interceptor = TrackingInterceptor()
        transport = AsyncHTTPTransport(interceptor=interceptor)

        mock_response = httpx.Response(200)

        async def mock_handle(*args, **kwargs):
            return mock_response

        mock_parent_handle = Mock(side_effect=mock_handle)

        original_handle = httpx.AsyncHTTPTransport.handle_async_request
        httpx.AsyncHTTPTransport.handle_async_request = mock_parent_handle

        try:
            # Make multiple async requests
            requests = [
                httpx.Request("GET", "https://api.example.com/1"),
                httpx.Request("POST", "https://api.example.com/2"),
                httpx.Request("DELETE", "https://api.example.com/3"),
            ]

            for req in requests:
                await transport.handle_async_request(req)

            # Verify all requests were intercepted
            assert len(interceptor.async_outbound_calls) == 3
            assert len(interceptor.async_inbound_calls) == 3
            assert interceptor.async_outbound_calls == requests
        finally:
            httpx.AsyncHTTPTransport.handle_async_request = original_handle
