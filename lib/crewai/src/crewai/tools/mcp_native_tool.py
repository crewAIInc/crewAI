"""Native MCP tool wrapper for CrewAI agents.

This module provides a tool wrapper that reuses existing MCP client sessions
for better performance and connection management.
"""

import asyncio
from typing import Any

from crewai.tools import BaseTool


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
    ) -> None:
        """Initialize native MCP tool.

        Args:
            mcp_client: MCPClient instance with active session.
            tool_name: Original name of the tool on the MCP server.
            tool_schema: Schema information for the tool.
            server_name: Name of the MCP server for prefixing.
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
        self._original_tool_name = tool_name
        self._server_name = server_name
        # self._logger = logging.getLogger(__name__)

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

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            Result from the MCP tool execution.
        """
        try:
            try:
                asyncio.get_running_loop()

                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    coro = self._run_async(**kwargs)
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            except RuntimeError:
                return asyncio.run(self._run_async(**kwargs))

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
        # Note: Since we use asyncio.run() which creates a new event loop each time,
        # Always reconnect on-demand because asyncio.run() creates new event loops per call
        # All MCP transport context managers (stdio, streamablehttp_client, sse_client)
        # use anyio.create_task_group() which can't span different event loops
        if self._mcp_client.connected:
            await self._mcp_client.disconnect()

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
                # Retry the call
                result = await self._mcp_client.call_tool(
                    self.original_tool_name, kwargs
                )
            else:
                raise

        finally:
            # Always disconnect after tool call to ensure clean context manager lifecycle
            # This prevents "exit cancel scope in different task" errors
            # All transport context managers must be exited in the same event loop they were entered
            await self._mcp_client.disconnect()

        # Extract result content
        if isinstance(result, str):
            return result

        # Handle various result formats
        if hasattr(result, "content") and result.content:
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    return str(content_item.text)
                return str(content_item)
            return str(result.content)

        return str(result)
