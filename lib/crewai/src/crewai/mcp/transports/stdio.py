"""Stdio transport for MCP servers running as local processes."""

import asyncio
import os
import subprocess
from typing import Any

from typing_extensions import Self

from crewai.mcp.transports.base import BaseTransport, TransportType


class StdioTransport(BaseTransport):
    """Stdio transport for connecting to local MCP servers.

    This transport connects to MCP servers running as local processes,
    communicating via standard input/output streams. Supports Python,
    Node.js, and other command-line servers.

    Example:
        ```python
        transport = StdioTransport(
            command="python",
            args=["path/to/server.py"],
            env={"API_KEY": "..."}
        )
        async with transport:
            # Use transport...
        ```
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize stdio transport.

        Args:
            command: Command to execute (e.g., "python", "node", "npx").
            args: Command arguments (e.g., ["server.py"] or ["-y", "@mcp/server"]).
            env: Environment variables to pass to the process.
            **kwargs: Additional transport options.
        """
        super().__init__(**kwargs)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._process: subprocess.Popen[bytes] | None = None
        self._transport_context: Any = None

    @property
    def transport_type(self) -> TransportType:
        """Return the transport type."""
        return TransportType.STDIO

    async def connect(self) -> Self:
        """Start the MCP server process and establish connection.

        Returns:
            Self for method chaining.

        Raises:
            ConnectionError: If process fails to start.
            ImportError: If MCP SDK not available.
        """
        if self._connected:
            return self

        try:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            process_env = os.environ.copy()
            process_env.update(self.env)

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=process_env if process_env else None,
            )
            self._transport_context = stdio_client(server_params)

            try:
                read, write = await self._transport_context.__aenter__()
            except Exception as e:
                import traceback

                traceback.print_exc()
                self._transport_context = None
                raise ConnectionError(
                    f"Failed to enter stdio transport context: {e}"
                ) from e

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
            raise ConnectionError(f"Failed to start MCP server process: {e}") from e

    async def disconnect(self) -> None:
        """Terminate the MCP server process and close connection."""
        if not self._connected:
            return

        try:
            self._clear_streams()

            if self._transport_context is not None:
                await self._transport_context.__aexit__(None, None, None)

            if self._process is not None:
                try:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self._process.kill()
                        await self._process.wait()
                # except ProcessLookupError:
                #     pass
                finally:
                    self._process = None

        except Exception as e:
            # Log but don't raise - cleanup should be best effort
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Error during stdio transport disconnect: {e}")

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
