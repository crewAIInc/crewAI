"""HTTP transport implementations for LLM request/response interception.

This module provides internal transport classes that integrate with BaseInterceptor
to enable request/response modification at the transport level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import httpx


if TYPE_CHECKING:
    from crewai.llms.hooks.base import BaseInterceptor


class HTTPTransport(httpx.HTTPTransport):
    """HTTP transport that uses an interceptor for request/response modification.

    This transport is used internally when a user provides a BaseInterceptor.
    Users should not instantiate this class directly - instead, pass an interceptor
    to the LLM client and this transport will be created automatically.
    """

    def __init__(
        self,
        interceptor: BaseInterceptor[Any],
        **kwargs: Any,
    ) -> None:
        """Initialize transport with interceptor.

        Args:
            interceptor: HTTP interceptor for modifying raw request/response objects.
            **kwargs: Additional arguments passed to httpx.HTTPTransport.
        """
        super().__init__(**kwargs)
        self.interceptor = interceptor

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle request with interception.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.
        """
        request = self.interceptor.on_outbound(request)
        response = super().handle_request(request)
        return cast(httpx.Response, self.interceptor.on_inbound(response))


class AsyncHTTPransport(httpx.AsyncHTTPTransport):
    """Async HTTP transport that uses an interceptor for request/response modification.

    This transport is used internally when a user provides a BaseInterceptor.
    Users should not instantiate this class directly - instead, pass an interceptor
    to the LLM client and this transport will be created automatically.
    """

    def __init__(
        self,
        interceptor: BaseInterceptor[Any],
        **kwargs: Any,
    ) -> None:
        """Initialize async transport with interceptor.

        Args:
            interceptor: HTTP interceptor for modifying raw request/response objects.
            **kwargs: Additional arguments passed to httpx.AsyncHTTPTransport.
        """
        super().__init__(**kwargs)
        self.interceptor = interceptor

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Handle async request with interception.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.
        """
        request = await self.interceptor.aon_outbound(request)
        response = await super().handle_async_request(request)
        return cast(httpx.Response, await self.interceptor.aon_inbound(response))
