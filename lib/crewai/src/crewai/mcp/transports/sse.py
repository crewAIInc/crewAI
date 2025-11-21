"""Server-Sent Events (SSE) transport for MCP servers."""

from typing import Any

from typing_extensions import Self

from crewai.mcp.transports.base import BaseTransport, TransportType


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
        ```
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SSE transport.

        Args:
            url: Server URL (e.g., "https://api.example.com/mcp/sse").
            headers: Optional HTTP headers.
            **kwargs: Additional transport options.
        """
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
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

            self._transport_context = sse_client(
                self.url,
                headers=self.headers if self.headers else None,
            )

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
