"""HTTP transport implementations for LLM request/response interception.

This module provides internal transport classes that integrate with BaseInterceptor
to enable request/response modification at the transport level.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict

from httpx import (
    AsyncHTTPTransport as _AsyncHTTPTransport,
    HTTPTransport as _HTTPTransport,
)
from typing_extensions import NotRequired, Unpack


if TYPE_CHECKING:
    from ssl import SSLContext

    from httpx import Limits, Request, Response
    from httpx._types import CertTypes, ProxyTypes

    from crewai.llms.hooks.base import BaseInterceptor


class HTTPTransportKwargs(TypedDict, total=False):
    """Typed dictionary for httpx.HTTPTransport initialization parameters.

    These parameters configure the underlying HTTP transport behavior including
    SSL verification, proxies, connection limits, and low-level socket options.
    """

    verify: bool | str | SSLContext
    cert: NotRequired[CertTypes]
    trust_env: bool
    http1: bool
    http2: bool
    limits: Limits
    proxy: NotRequired[ProxyTypes]
    uds: NotRequired[str]
    local_address: NotRequired[str]
    retries: int
    socket_options: NotRequired[
        Iterable[
            tuple[int, int, int]
            | tuple[int, int, bytes | bytearray]
            | tuple[int, int, None, int]
        ]
    ]


class HTTPTransport(_HTTPTransport):
    """HTTP transport that uses an interceptor for request/response modification.

    This transport is used internally when a user provides a BaseInterceptor.
    Users should not instantiate this class directly - instead, pass an interceptor
    to the LLM client and this transport will be created automatically.
    """

    def __init__(
        self,
        interceptor: BaseInterceptor[Request, Response],
        **kwargs: Unpack[HTTPTransportKwargs],
    ) -> None:
        """Initialize transport with interceptor.

        Args:
            interceptor: HTTP interceptor for modifying raw request/response objects.
            **kwargs: HTTPTransport configuration parameters (verify, cert, proxy, etc.).
        """
        super().__init__(**kwargs)
        self.interceptor = interceptor

    def handle_request(self, request: Request) -> Response:
        """Handle request with interception.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.
        """
        request = self.interceptor.on_outbound(request)
        response = super().handle_request(request)
        return self.interceptor.on_inbound(response)


class AsyncHTTPTransport(_AsyncHTTPTransport):
    """Async HTTP transport that uses an interceptor for request/response modification.

    This transport is used internally when a user provides a BaseInterceptor.
    Users should not instantiate this class directly - instead, pass an interceptor
    to the LLM client and this transport will be created automatically.
    """

    def __init__(
        self,
        interceptor: BaseInterceptor[Request, Response],
        **kwargs: Unpack[HTTPTransportKwargs],
    ) -> None:
        """Initialize async transport with interceptor.

        Args:
            interceptor: HTTP interceptor for modifying raw request/response objects.
            **kwargs: HTTPTransport configuration parameters (verify, cert, proxy, etc.).
        """
        super().__init__(**kwargs)
        self.interceptor = interceptor

    async def handle_async_request(self, request: Request) -> Response:
        """Handle async request with interception.

        Args:
            request: The HTTP request to handle.

        Returns:
            The HTTP response.
        """
        request = await self.interceptor.aon_outbound(request)
        response = await super().handle_async_request(request)
        return await self.interceptor.aon_inbound(response)
