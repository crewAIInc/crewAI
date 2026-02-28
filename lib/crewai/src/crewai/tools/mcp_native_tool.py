"""Native MCP tool wrapper for CrewAI agents.

This module provides a tool wrapper that reuses existing MCP client sessions
for better performance and connection management.
"""

import asyncio
import threading
from typing import Any

from crewai.tools import BaseTool


_mcp_loop: asyncio.AbstractEventLoop | None = None
_mcp_loop_thread: threading.Thread | None = None
_mcp_loop_lock = threading.Lock()


def _get_mcp_event_loop() -> asyncio.AbstractEventLoop:
    """Return (and lazily start) a persistent event loop for MCP operations.

    All MCP SDK transports use anyio task groups whose cancel scopes must be
    entered and exited on the same event loop / task.  By funnelling every
    MCP call through one long-lived loop we avoid the "exit cancel scope in
    a different task" errors that happen when asyncio.run() creates a
    throwaway loop per call.
    """
    global _mcp_loop, _mcp_loop_thread
    with _mcp_loop_lock:
        if _mcp_loop is None or _mcp_loop.is_closed():
            _mcp_loop = asyncio.new_event_loop()
            _mcp_loop_thread = threading.Thread(
                target=_mcp_loop.run_forever, daemon=True, name="mcp-event-loop"
            )
            _mcp_loop_thread.start()
    return _mcp_loop


class MCPNativeTool(BaseTool):
    """Native MCP tool that reuses client sessions.

    This tool wrapper is used when agents connect to MCP servers using
    structured configurations. It reuses existing client sessions for
    better performance and proper connection lifecycle management.

    Unlike MCPToolWrapper which connects on-demand, this tool uses
    a shared MCP client instance that maintains a persistent connection.
    """

    def __init__(
        self,
        mcp_client: Any,
        tool_name: str,
        tool_schema: dict[str, Any],
        server_name: str,
        original_tool_name: str | None = None,
    ) -> None:
        """Initialize native MCP tool.

        Args:
            mcp_client: MCPClient instance with active session.
            tool_name: Name of the tool (may be prefixed).
            tool_schema: Schema information for the tool.
            server_name: Name of the MCP server for prefixing.
            original_tool_name: Original name of the tool on the MCP server.
        """
        prefixed_name = f"{server_name}_{tool_name}"

        args_schema = tool_schema.get("args_schema")

        kwargs = {
            "name": prefixed_name,
            "description": tool_schema.get(
                "description", f"Tool {tool_name} from {server_name}"
            ),
        }

        if args_schema is not None:
            kwargs["args_schema"] = args_schema

        super().__init__(**kwargs)

        self._mcp_client = mcp_client
        self._original_tool_name = original_tool_name or tool_name
        self._server_name = server_name

    @property
    def mcp_client(self) -> Any:
        """Get the MCP client instance."""
        return self._mcp_client

    @property
    def original_tool_name(self) -> str:
        """Get the original tool name."""
        return self._original_tool_name

    @property
    def server_name(self) -> str:
        """Get the server name."""
        return self._server_name

    def _run(self, **kwargs) -> str:
        """Execute tool using the MCP client session.

        Submits work to a persistent background event loop so that all MCP
        transport context managers (which rely on anyio cancel scopes) stay
        on the same loop and task throughout their lifecycle.

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            Result from the MCP tool execution.
        """
        loop = _get_mcp_event_loop()
        timeout = self._mcp_client.connect_timeout + self._mcp_client.execution_timeout
        try:
            future = asyncio.run_coroutine_threadsafe(self._run_async(**kwargs), loop)
            return future.result(timeout=timeout)
        except Exception as e:
            raise RuntimeError(
                f"Error executing MCP tool {self.original_tool_name}: {e!s}"
            ) from e

    async def _run_async(self, **kwargs) -> str:
        """Async implementation of tool execution.

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            Result from the MCP tool execution.
        """
        if not self._mcp_client.connected:
            await self._mcp_client.connect()

        try:
            result = await self._mcp_client.call_tool(self.original_tool_name, kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if (
                "not connected" in error_str
                or "connection" in error_str
                or "send" in error_str
            ):
                await self._mcp_client.disconnect()
                await self._mcp_client.connect()
                result = await self._mcp_client.call_tool(
                    self.original_tool_name, kwargs
                )
            else:
                raise

        if isinstance(result, str):
            return result

        if hasattr(result, "content") and result.content:
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    return str(content_item.text)
                return str(content_item)
            return str(result.content)

        return str(result)
