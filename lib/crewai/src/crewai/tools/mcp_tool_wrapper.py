"""MCP Tool Wrapper for on-demand MCP server connections."""

import asyncio
from collections.abc import Callable
from typing import Any

from crewai.tools import BaseTool


MCP_CONNECTION_TIMEOUT = 15
MCP_TOOL_EXECUTION_TIMEOUT = 60
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
        progress_callback: Callable[[float, float | None, str | None], None] | None = None,
        agent: Any | None = None,
        task: Any | None = None,
    ):
        """Initialize the MCP tool wrapper.

        Args:
            mcp_server_params: Parameters for connecting to the MCP server
            tool_name: Original name of the tool on the MCP server
            tool_schema: Schema information for the tool
            server_name: Name of the MCP server for prefixing
            progress_callback: Optional callback for progress notifications (progress, total, message)
            agent: Optional agent context for event emission
            task: Optional task context for event emission
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
        self._progress_callback = progress_callback
        self._agent = agent
        self._task = task

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
            return f"Error executing MCP tool {self.original_tool_name}: {e!s}"

    async def _run_async(self, **kwargs) -> str:
        """Async implementation of MCP tool execution with timeouts and retry logic."""
        return await self._retry_with_exponential_backoff(
            self._execute_tool_with_timeout, **kwargs
        )

    async def _retry_with_exponential_backoff(self, operation_func, **kwargs) -> str:
        """Retry operation with exponential backoff, avoiding try-except in loop for performance."""
        last_error = None

        for attempt in range(MCP_MAX_RETRIES):
            # Execute single attempt outside try-except loop structure
            result, error, should_retry = await self._execute_single_attempt(
                operation_func, **kwargs
            )

            # Success case - return immediately
            if result is not None:
                return result

            # Non-retryable error - return immediately
            if not should_retry:
                return error

            # Retryable error - continue with backoff
            last_error = error
            if attempt < MCP_MAX_RETRIES - 1:
                wait_time = 2**attempt  # Exponential backoff
                await asyncio.sleep(wait_time)

        return (
            f"MCP tool execution failed after {MCP_MAX_RETRIES} attempts: {last_error}"
        )

    async def _execute_single_attempt(
        self, operation_func, **kwargs
    ) -> tuple[str | None, str, bool]:
        """Execute single operation attempt and return (result, error_message, should_retry)."""
        try:
            result = await operation_func(**kwargs)
            return result, "", False

        except ImportError:
            return (
                None,
                "MCP library not available. Please install with: pip install mcp",
                False,
            )

        except asyncio.TimeoutError:
            return (
                None,
                f"Connection timed out after {MCP_TOOL_EXECUTION_TIMEOUT} seconds",
                True,
            )

        except Exception as e:
            error_str = str(e).lower()

            # Classify errors as retryable or non-retryable
            if "authentication" in error_str or "unauthorized" in error_str:
                return None, f"Authentication failed for MCP server: {e!s}", False
            if "not found" in error_str:
                return (
                    None,
                    f"Tool '{self.original_tool_name}' not found on MCP server",
                    False,
                )
            if "connection" in error_str or "network" in error_str:
                return None, f"Network connection failed: {e!s}", True
            if "json" in error_str or "parsing" in error_str:
                return None, f"Server response parsing error: {e!s}", True
            return None, f"MCP execution error: {e!s}", False

    async def _execute_tool_with_timeout(self, **kwargs) -> str:
        """Execute tool with timeout wrapper."""
        return await asyncio.wait_for(
            self._execute_tool(**kwargs), timeout=MCP_TOOL_EXECUTION_TIMEOUT
        )

    async def _execute_tool(self, **kwargs) -> str:
        """Execute the actual MCP tool call with progress support."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        server_url = self.mcp_server_params["url"]
        headers = self.mcp_server_params.get("headers")

        try:
            # Wrap entire operation with single timeout
            async def _do_mcp_call():
                client_kwargs = {"terminate_on_close": True}
                if headers:
                    client_kwargs["headers"] = headers

                async with streamablehttp_client(
                    server_url, **client_kwargs
                ) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()

                        # Register progress handler if callback is provided
                        if self._progress_callback:
                            def progress_handler(progress_notification):
                                """Handle progress notifications from MCP server."""
                                progress = progress_notification.progress
                                total = getattr(progress_notification, "total", None)
                                message = getattr(progress_notification, "message", None)
                                
                                self._progress_callback(progress, total, message)
                                
                                self._emit_progress_event(progress, total, message)

                            session.on_progress = progress_handler

                        result = await session.call_tool(
                            self.original_tool_name, kwargs
                        )

                        # Extract the result content
                        if hasattr(result, "content") and result.content:
                            if (
                                isinstance(result.content, list)
                                and len(result.content) > 0
                            ):
                                content_item = result.content[0]
                                if hasattr(content_item, "text"):
                                    return str(content_item.text)
                                return str(content_item)
                            return str(result.content)
                        return str(result)

            return await asyncio.wait_for(
                _do_mcp_call(), timeout=MCP_TOOL_EXECUTION_TIMEOUT
            )

        except asyncio.CancelledError as e:
            raise asyncio.TimeoutError("MCP operation was cancelled") from e
        except Exception as e:
            if hasattr(e, "__cause__") and e.__cause__:
                raise asyncio.TimeoutError(
                    f"MCP connection error: {e.__cause__}"
                ) from e.__cause__

            if "TaskGroup" in str(e) or "unhandled errors" in str(e):
                raise asyncio.TimeoutError(f"MCP connection error: {e}") from e
            raise

    def _emit_progress_event(
        self, progress: float, total: float | None, message: str | None
    ) -> None:
        """Emit MCPToolProgressEvent to CrewAI event bus."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.tool_usage_events import MCPToolProgressEvent

        event_data = {
            "tool_name": self.original_tool_name,
            "server_name": self.server_name,
            "progress": progress,
            "total": total,
            "message": message,
        }

        if self._agent:
            event_data["agent_id"] = str(self._agent.id) if hasattr(self._agent, "id") else None
            event_data["agent_role"] = getattr(self._agent, "role", None)

        if self._task:
            event_data["task_id"] = str(self._task.id) if hasattr(self._task, "id") else None
            event_data["task_name"] = getattr(self._task, "name", None) or getattr(self._task, "description", None)

        event = MCPToolProgressEvent(**event_data)
        crewai_event_bus.emit(self, event)
