"""HTTP and Streamable HTTP transport for MCP servers."""

import asyncio
from typing import Any

from typing_extensions import Self

from crewai.mcp._utils import async_timeout
from crewai.mcp.transports.base import BaseTransport, TransportType


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
            ConnectionError: If connection fails.
            ImportError: If MCP SDK not available.
        """
        if self._connected:
            return self

        try:
            from mcp.client.streamable_http import streamablehttp_client

            self._transport_context = streamablehttp_client(
                self.url,
                headers=self.headers if self.headers else None,
                terminate_on_close=True,
            )

            try:
                # Enter and exit the SDK's anyio cancel scope in the same task.
                # asyncio.wait_for() runs __aenter__ in a child task and can
                # later trigger "cancel scope in a different task" failures.
                async with async_timeout(self.connect_timeout):
                    read, write, _ = await self._transport_context.__aenter__()
            # async-timeout raises asyncio.TimeoutError on Python 3.10, where
            # it is distinct from the built-in TimeoutError.
            except (TimeoutError, asyncio.TimeoutError) as e:
                self._transport_context = None
                raise ConnectionError(
                    f"Transport context entry timed out after {self.connect_timeout} seconds. "
                    "Server may be slow or unreachable."
                ) from e
            except Exception as e:
                self._transport_context = None
                raise ConnectionError(f"Failed to enter transport context: {e}") from e
            self._set_streams(read=read, write=write)
            return self

        except ImportError as e:
            raise ImportError(
                "MCP library not available. Please install with: pip install mcp"
            ) from e
        except Exception as e:
            self._clear_streams()
            if self._transport_context is not None:
                self._transport_context = None
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    async def _disconnect(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: Any = None,
    ) -> None:
        """Close the SDK context with the exception that triggered unwinding."""
        if not self._connected:
            return

        self._clear_streams()
        transport_context = self._transport_context
        self._transport_context = None
        if transport_context is not None:
            await transport_context.__aexit__(exc_type, exc_val, exc_tb)

    async def disconnect(self) -> None:
        """Close HTTP connection."""
        await self._disconnect()

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
        await self._disconnect(exc_type, exc_val, exc_tb)
