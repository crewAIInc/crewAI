"""Server-Sent Events (SSE) transport for MCP servers."""

from typing import Any

import httpx
from typing_extensions import Self

from crewai.mcp.transports.base import BaseTransport, TransportType


def _create_httpx_client_factory(
    verify: bool | str,
) -> Any:
    """Create a custom httpx client factory with SSL verification settings.

    This factory preserves MCP's default client settings (follow_redirects, timeout)
    while allowing customization of SSL verification.

    Args:
        verify: SSL verification setting. True for default verification,
                False to disable, or a path to a CA bundle file.

    Returns:
        A factory function compatible with MCP's McpHttpClientFactory protocol.
    """

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "follow_redirects": True,
            "verify": verify,
        }

        if timeout is None:
            kwargs["timeout"] = httpx.Timeout(30.0)
        else:
            kwargs["timeout"] = timeout

        if headers is not None:
            kwargs["headers"] = headers

        if auth is not None:
            kwargs["auth"] = auth

        return httpx.AsyncClient(**kwargs)

    return factory


class SSETransport(BaseTransport):
    """SSE transport for connecting to remote MCP servers.

    This transport connects to MCP servers using Server-Sent Events (SSE)
    for real-time streaming communication.

    Example:
        ```python
        transport = SSETransport(
            url="https://api.example.com/mcp/sse",
            headers={"Authorization": "Bearer ..."}
        )
        async with transport:
            # Use transport...

        # With SSL verification disabled
        transport = SSETransport(
            url="https://internal-server.example.com/mcp/sse",
            verify=False
        )
        ```
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        verify: bool | str = True,
        **kwargs: Any,
    ) -> None:
        """Initialize SSE transport.

        Args:
            url: Server URL (e.g., "https://api.example.com/mcp/sse").
            headers: Optional HTTP headers.
            verify: SSL certificate verification. Set to False to disable,
                    or provide a path to a CA bundle file (default: True).
            **kwargs: Additional transport options.
        """
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self.verify = verify
        self._transport_context: Any = None

    @property
    def transport_type(self) -> TransportType:
        """Return the transport type."""
        return TransportType.SSE

    async def connect(self) -> Self:
        """Establish SSE connection to MCP server.

        Returns:
            Self for method chaining.

        Raises:
            ConnectionError: If connection fails.
            ImportError: If MCP SDK not available.
        """
        if self._connected:
            return self

        try:
            from mcp.client.sse import sse_client

            client_kwargs: dict[str, Any] = {
                "headers": self.headers if self.headers else None,
            }

            if self.verify is not True:
                client_kwargs["httpx_client_factory"] = _create_httpx_client_factory(
                    self.verify
                )

            self._transport_context = sse_client(self.url, **client_kwargs)

            read, write = await self._transport_context.__aenter__()

            self._set_streams(read=read, write=write)

            return self

        except ImportError as e:
            raise ImportError(
                "MCP library not available. Please install with: pip install mcp"
            ) from e
        except Exception as e:
            self._clear_streams()
            raise ConnectionError(f"Failed to connect to SSE MCP server: {e}") from e

    async def disconnect(self) -> None:
        """Close SSE connection."""
        if not self._connected:
            return

        try:
            self._clear_streams()
            if self._transport_context is not None:
                await self._transport_context.__aexit__(None, None, None)

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Error during SSE transport disconnect: {e}")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()
