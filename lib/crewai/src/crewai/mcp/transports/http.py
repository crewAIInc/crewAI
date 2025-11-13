"""HTTP and Streamable HTTP transport for MCP servers."""

import asyncio
from typing import Any

from typing_extensions import Self


# BaseExceptionGroup is available in Python 3.11+
try:
    from builtins import BaseExceptionGroup
except ImportError:
    # Fallback for Python < 3.11 (shouldn't happen in practice)
    BaseExceptionGroup = Exception

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

    async def disconnect(self) -> None:
        """Close HTTP connection."""
        if not self._connected:
            return

        try:
            # Clear streams first
            self._clear_streams()
            # await self._exit_stack.aclose()

            # Exit transport context - this will clean up background tasks
            # Give a small delay to allow background tasks to complete
            if self._transport_context is not None:
                try:
                    # Wait a tiny bit for any pending operations
                    await asyncio.sleep(0.1)
                    await self._transport_context.__aexit__(None, None, None)
                except (RuntimeError, asyncio.CancelledError) as e:
                    # Ignore "exit cancel scope in different task" errors and cancellation
                    # These happen when asyncio.run() closes the event loop
                    # while background tasks are still running
                    error_msg = str(e).lower()
                    if "cancel scope" not in error_msg and "task" not in error_msg:
                        # Only suppress cancel scope/task errors, re-raise others
                        if isinstance(e, RuntimeError):
                            raise
                        # For CancelledError, just suppress it
                except BaseExceptionGroup as eg:
                    # Handle exception groups from anyio task groups
                    # Suppress if they contain cancel scope errors
                    should_suppress = False
                    for exc in eg.exceptions:
                        error_msg = str(exc).lower()
                        if "cancel scope" in error_msg or "task" in error_msg:
                            should_suppress = True
                            break
                    if not should_suppress:
                        raise
                except Exception as e:
                    raise RuntimeError(
                        f"Error during HTTP transport disconnect: {e}"
                    ) from e

            self._connected = False

        except Exception as e:
            # Log but don't raise - cleanup should be best effort
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Error during HTTP transport disconnect: {e}")

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
