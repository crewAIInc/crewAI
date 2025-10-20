"""MCP Tool Wrapper for on-demand MCP server connections."""

import asyncio
import time
from typing import Any

from crewai.tools import BaseTool

# MCP Connection timeout constants (in seconds)
MCP_CONNECTION_TIMEOUT = 10
MCP_TOOL_EXECUTION_TIMEOUT = 30
MCP_DISCOVERY_TIMEOUT = 15
MCP_MAX_RETRIES = 3


class MCPToolWrapper(BaseTool):
    """Lightweight wrapper for MCP tools that connects on-demand."""

    def __init__(
        self,
        mcp_server_params: dict,
        tool_name: str,
        tool_schema: dict,
        server_name: str,
    ):
        """Initialize the MCP tool wrapper.

        Args:
            mcp_server_params: Parameters for connecting to the MCP server
            tool_name: Original name of the tool on the MCP server
            tool_schema: Schema information for the tool
            server_name: Name of the MCP server for prefixing
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
        self._mcp_server_params = mcp_server_params
        self._original_tool_name = tool_name
        self._server_name = server_name

    @property
    def mcp_server_params(self) -> dict:
        """Get the MCP server parameters."""
        return self._mcp_server_params

    @property
    def original_tool_name(self) -> str:
        """Get the original tool name."""
        return self._original_tool_name

    @property
    def server_name(self) -> str:
        """Get the server name."""
        return self._server_name

    def _run(self, **kwargs) -> str:
        """Connect to MCP server and execute tool.

        Args:
            **kwargs: Arguments to pass to the MCP tool

        Returns:
            Result from the MCP tool execution
        """
        try:
            return asyncio.run(self._run_async(**kwargs))
        except asyncio.TimeoutError:
            return f"MCP tool '{self.original_tool_name}' timed out after {MCP_TOOL_EXECUTION_TIMEOUT} seconds"
        except Exception as e:
            return f"Error executing MCP tool {self.original_tool_name}: {str(e)}"

    async def _run_async(self, **kwargs) -> str:
        """Async implementation of MCP tool execution with timeouts and retry logic."""
        last_error = None

        for attempt in range(MCP_MAX_RETRIES):
            try:
                result = await asyncio.wait_for(
                    self._execute_tool(**kwargs),
                    timeout=MCP_TOOL_EXECUTION_TIMEOUT
                )
                return result

            except asyncio.TimeoutError:
                last_error = f"Connection timed out after {MCP_TOOL_EXECUTION_TIMEOUT} seconds"
                if attempt < MCP_MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

            except ImportError:
                return "MCP library not available. Please install with: pip install mcp"

            except Exception as e:
                error_str = str(e).lower()

                # Handle specific error types
                if 'connection' in error_str or 'network' in error_str:
                    last_error = f"Network connection failed: {str(e)}"
                elif 'authentication' in error_str or 'unauthorized' in error_str:
                    return f"Authentication failed for MCP server: {str(e)}"
                elif 'json' in error_str or 'parsing' in error_str:
                    last_error = f"Server response parsing error: {str(e)}"
                elif 'not found' in error_str:
                    return f"Tool '{self.original_tool_name}' not found on MCP server"
                else:
                    last_error = f"MCP execution error: {str(e)}"

                # Retry for transient errors
                if attempt < MCP_MAX_RETRIES - 1 and ('connection' in error_str or 'network' in error_str or 'json' in error_str):
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break

        return f"MCP tool execution failed after {MCP_MAX_RETRIES} attempts: {last_error}"

    async def _execute_tool(self, **kwargs) -> str:
        """Execute the actual MCP tool call."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        server_url = self.mcp_server_params["url"]

        # Connect to MCP server with timeout
        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Initialize the connection with timeout
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=MCP_CONNECTION_TIMEOUT
                )

                # Call the specific tool with timeout
                result = await asyncio.wait_for(
                    session.call_tool(self.original_tool_name, kwargs),
                    timeout=MCP_TOOL_EXECUTION_TIMEOUT - MCP_CONNECTION_TIMEOUT
                )

                # Extract the result content
                if hasattr(result, 'content') and result.content:
                    if isinstance(result.content, list) and len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, 'text'):
                            return str(content_item.text)
                        else:
                            return str(content_item)
                    else:
                        return str(result.content)
                else:
                    return str(result)
