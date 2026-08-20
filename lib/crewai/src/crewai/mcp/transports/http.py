"""HTTP and Streamable HTTP transport for MCP servers."""

import asyncio
import contextlib
import logging
import sys
from typing import Any

from typing_extensions import Self


if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup

from crewai.mcp.exceptions import (
    error_for_status,
    find_http_status,
    find_transport_failure,
    raise_connection_failure,
)
from crewai.mcp.transports.base import BaseTransport, TransportType


logger = logging.getLogger(__name__)


class HTTPTransport(BaseTransport):
    """HTTP/Streamable HTTP transport for connecting to remote MCP servers.

    This transport connects to MCP servers over HTTP/HTTPS using the
    streamable HTTP client from the MCP SDK.

    Example:
        ```python
        transport = HTTPTransport(
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer ..."}
        )
        async with transport:
            # Use transport...
        ```
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        streamable: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize HTTP transport.

        Args:
            url: Server URL (e.g., "https://api.example.com/mcp").
            headers: Optional HTTP headers.
            streamable: Whether to use streamable HTTP (default: True).
            **kwargs: Additional transport options.
        """
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self.streamable = streamable
        self._transport_context: Any = None

    @property
    def transport_type(self) -> TransportType:
        """Return the transport type."""
        return TransportType.STREAMABLE_HTTP if self.streamable else TransportType.HTTP

    async def connect(self) -> Self:
        """Establish HTTP connection to MCP server.

        Returns:
            Self for method chaining.

        Raises:
            MCPConnectionError: If the server refused the connection with an HTTP status.
            ConnectionError: If connection fails for other reasons.
            ImportError: If MCP SDK not available.
        """
        if self._connected:
            return self

        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:
            raise ImportError(
                "MCP library not available. Please install with: pip install mcp"
            ) from e

        self._transport_context = streamablehttp_client(
            self.url,
            headers=self.headers if self.headers else None,
            terminate_on_close=True,
        )

        try:
            read, write, _ = await asyncio.wait_for(
                self._transport_context.__aenter__(), timeout=30.0
            )
        except asyncio.TimeoutError as e:
            self._transport_context = None
            raise ConnectionError(
                "Transport context entry timed out after 30 seconds. "
                "Server may be slow or unreachable."
            ) from e
        except Exception as e:
            self._clear_streams()
            self._transport_context = None
            raise_connection_failure(f"Failed to connect to MCP server: {e}", e)

        self._set_streams(read=read, write=write)
        return self

    async def disconnect(self) -> None:
        """Close HTTP connection.

        Raises:
            MCPConnectionError: If the server refused the connection. The
                refusal is reported here rather than to the coroutine that
                was waiting, because the underlying task group only raises it
                while unwinding.
        """
        if not self._connected:
            return

        self._clear_streams()
        context, self._transport_context = self._transport_context, None
        self._connected = False

        if context is None:
            return

        try:
            # Give pending background operations a moment to finish. Unwinding
            # still has to happen if this is cancelled, since it is the only
            # thing that reveals why the connection failed.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0.1)
            await context.__aexit__(None, None, None)
        except asyncio.CancelledError:
            logger.debug("MCP HTTP transport teardown was cancelled")
        except (Exception, BaseExceptionGroup) as e:
            if (status_code := find_http_status(e)) is not None:
                raise error_for_status(status_code) from e
            if find_transport_failure(e) is not None:
                raise
            # Unwinding an anyio task group routinely reports cancel-scope and
            # async-generator errors that describe the teardown itself rather
            # than the failure that caused it.
            logger.debug("Ignoring MCP HTTP transport teardown error: %s", e)

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
