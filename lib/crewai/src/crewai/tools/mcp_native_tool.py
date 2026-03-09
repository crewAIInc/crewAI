"""Native MCP tool wrapper for CrewAI agents.

This module provides a tool wrapper that reuses existing MCP client sessions
for better performance and connection management.
"""

import asyncio
import logging
import threading
from typing import Any

from crewai.tools import BaseTool


logger = logging.getLogger(__name__)


_mcp_loop_lock = threading.Lock()
_mcp_shared_loop: asyncio.AbstractEventLoop | None = None
_mcp_shared_loop_thread: threading.Thread | None = None


class MCPNativeTool(BaseTool):
    """Native MCP tool that reuses client sessions.

    This tool wrapper is used when agents connect to MCP servers using
    structured configurations. It reuses existing client sessions for
    better performance and proper connection lifecycle management.

    A dedicated background event loop is used for all MCP operations so that
    anyio cancel scopes (used by streamable-HTTP and SSE transports) are
    always entered and exited in the same async task.  This avoids the
    ``RuntimeError: Attempted to exit cancel scope in a different task``
    that occurs when ``asyncio.run()`` creates throwaway event loops.
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
        # Create tool name with server prefix to avoid conflicts
        prefixed_name = f"{server_name}_{tool_name}"

        # Handle args_schema properly - BaseTool expects a BaseModel subclass
        args_schema = tool_schema.get("args_schema")

        # Only pass args_schema if it's provided
        kwargs = {
            "name": prefixed_name,
            "description": tool_schema.get(
                "description", f"Tool {tool_name} from {server_name}"
            ),
        }

        if args_schema is not None:
            kwargs["args_schema"] = args_schema

        super().__init__(**kwargs)

        # Set instance attributes after super().__init__
        self._mcp_client = mcp_client
        self._original_tool_name = original_tool_name or tool_name
        self._server_name = server_name



    @staticmethod
    def _ensure_loop() -> asyncio.AbstractEventLoop:
        """Return a dedicated event loop running in a background thread.

        The loop is shared across all MCPNativeTool instances so that MCP
        connections from the same process coexist on a single loop, which
        is both resource-efficient and avoids cross-loop cancel-scope issues.
        """
        global _mcp_shared_loop, _mcp_shared_loop_thread

        if _mcp_shared_loop is not None and _mcp_shared_loop.is_running():
            return _mcp_shared_loop

        with _mcp_loop_lock:
            # Double-check after acquiring the lock
            if _mcp_shared_loop is not None and _mcp_shared_loop.is_running():
                return _mcp_shared_loop

            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                daemon=True,
                name="mcp-native-tool-loop",
            )
            thread.start()
            _mcp_shared_loop = loop
            _mcp_shared_loop_thread = thread
            return loop

    def _run_in_dedicated_loop(self, coro: Any) -> Any:
        """Submit *coro* to the dedicated loop and block until it completes."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


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

        All async work is dispatched to a long-lived background event loop
        so that transport-level cancel scopes stay within a single async
        task, regardless of whether the caller is inside a Flow thread,
        another event loop, or plain synchronous code.
        """
        try:
            return self._run_in_dedicated_loop(self._run_async(**kwargs))
        except Exception as e:
            raise RuntimeError(
                f"Error executing MCP tool {self.original_tool_name}: {e!s}"
            ) from e

    async def _run_async(self, **kwargs) -> str:
        """Async implementation of tool execution.

        Each invocation owns its full connect → call → disconnect lifecycle
        so that anyio cancel scopes are always entered and exited in the
        same asyncio Task (run_coroutine_threadsafe creates a new Task per
        submission).
        """
        # Always start with a fresh connection in THIS task.
        if self._mcp_client.connected:
            try:
                await self._mcp_client.disconnect()
            except Exception:
                logger.debug("Failed to disconnect stale MCP client", exc_info=True)

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
                # Connection broke mid-call — reconnect and retry once
                try:
                    await self._mcp_client.disconnect()
                except Exception:
                    logger.debug("Failed to disconnect MCP client during retry", exc_info=True)
                await self._mcp_client.connect()
                result = await self._mcp_client.call_tool(
                    self.original_tool_name, kwargs
                )
            else:
                raise
        finally:
            try:
                await self._mcp_client.disconnect()
            except Exception:
                logger.debug("Failed to disconnect MCP client during cleanup", exc_info=True)

        # Extract result content
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
