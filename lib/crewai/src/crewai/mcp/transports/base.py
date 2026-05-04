"""Base transport interface for MCP connections."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from typing_extensions import Self


MCPReadStream = MemoryObjectReceiveStream[SessionMessage | Exception]
MCPWriteStream = MemoryObjectSendStream[SessionMessage]


class TransportType(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"
    HTTP = "http"
    STREAMABLE_HTTP = "streamable-http"
    SSE = "sse"


class BaseTransport(ABC):
    """Base class for MCP transport implementations.

    This abstract base class defines the interface that all transport
    implementations must follow. Transports handle the low-level communication
    with MCP servers.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the transport.

        Args:
            **kwargs: Transport-specific configuration options.
        """
        self._read_stream: MCPReadStream | None = None
        self._write_stream: MCPWriteStream | None = None
        self._connected = False

    @property
    @abstractmethod
    def transport_type(self) -> TransportType:
        """Return the transport type."""
        ...

    @property
    def connected(self) -> bool:
        """Check if transport is connected."""
        return self._connected

    @property
    def read_stream(self) -> MCPReadStream:
        """Get the read stream."""
        if self._read_stream is None:
            raise RuntimeError("Transport not connected. Call connect() first.")
        return self._read_stream

    @property
    def write_stream(self) -> MCPWriteStream:
        """Get the write stream."""
        if self._write_stream is None:
            raise RuntimeError("Transport not connected. Call connect() first.")
        return self._write_stream

    @abstractmethod
    async def connect(self) -> Self:
        """Establish connection to MCP server.

        Returns:
            Self for method chaining.

        Raises:
            ConnectionError: If connection fails.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        ...

    def _set_streams(self, read: MCPReadStream, write: MCPWriteStream) -> None:
        """Set the read and write streams.

        Args:
            read: Read stream.
            write: Write stream.
        """
        self._read_stream = read
        self._write_stream = write
        self._connected = True

    def _clear_streams(self) -> None:
        """Clear the read and write streams."""
        self._read_stream = None
        self._write_stream = None
        self._connected = False
