"""MCP client with session management for CrewAI agents."""

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import datetime
import logging
import time
from typing import Any

from typing_extensions import Self


# BaseExceptionGroup is available in Python 3.11+
try:
    from builtins import BaseExceptionGroup
except ImportError:
    # Fallback for Python < 3.11 (shouldn't happen in practice)
    BaseExceptionGroup = Exception

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.mcp_events import (
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
)
from crewai.mcp.transports.base import BaseTransport
from crewai.mcp.transports.http import HTTPTransport
from crewai.mcp.transports.sse import SSETransport
from crewai.mcp.transports.stdio import StdioTransport
from crewai.utilities.string_utils import sanitize_tool_name


# MCP Connection timeout constants (in seconds)
MCP_CONNECTION_TIMEOUT = 30  # Increased for slow servers
MCP_TOOL_EXECUTION_TIMEOUT = 30
MCP_DISCOVERY_TIMEOUT = 30  # Increased for slow servers
MCP_MAX_RETRIES = 3

# Simple in-memory cache for MCP tool schemas (duration: 5 minutes)
_mcp_schema_cache: dict[str, tuple[dict[str, Any], float]] = {}
_cache_ttl = 300  # 5 minutes


class MCPClient:
    """MCP client with session management.

    This client manages connections to MCP servers and provides a high-level
    interface for interacting with MCP tools, prompts, and resources.

    Example:
        ```python
        transport = StdioTransport(command="python", args=["server.py"])
        client = MCPClient(transport)
        async with client:
            tools = await client.list_tools()
            result = await client.call_tool("tool_name", {"arg": "value"})
        ```
    """

    def __init__(
        self,
        transport: BaseTransport,
        connect_timeout: int = MCP_CONNECTION_TIMEOUT,
        execution_timeout: int = MCP_TOOL_EXECUTION_TIMEOUT,
        discovery_timeout: int = MCP_DISCOVERY_TIMEOUT,
        max_retries: int = MCP_MAX_RETRIES,
        cache_tools_list: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize MCP client.

        Args:
            transport: Transport instance for MCP server connection.
            connect_timeout: Connection timeout in seconds.
            execution_timeout: Tool execution timeout in seconds.
            discovery_timeout: Tool discovery timeout in seconds.
            max_retries: Maximum retry attempts for operations.
            cache_tools_list: Whether to cache tool list results.
            logger: Optional logger instance.
        """
        self.transport = transport
        self.connect_timeout = connect_timeout
        self.execution_timeout = execution_timeout
        self.discovery_timeout = discovery_timeout
        self.max_retries = max_retries
        self.cache_tools_list = cache_tools_list
        # self._logger = logger or logging.getLogger(__name__)
        self._session: Any = None
        self._initialized = False
        self._exit_stack = AsyncExitStack()
        self._was_connected = False

    @property
    def connected(self) -> bool:
        """Check if client is connected to server."""
        return self.transport.connected and self._initialized

    @property
    def session(self) -> Any:
        """Get the MCP session."""
        if self._session is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._session

    def _get_server_info(self) -> tuple[str, str | None, str | None]:
        """Get server information for events.

        Returns:
            Tuple of (server_name, server_url, transport_type).
        """
        if isinstance(self.transport, StdioTransport):
            server_name = f"{self.transport.command} {' '.join(self.transport.args)}"
            server_url = None
            transport_type = self.transport.transport_type.value
        elif isinstance(self.transport, HTTPTransport):
            server_name = self.transport.url
            server_url = self.transport.url
            transport_type = self.transport.transport_type.value
        elif isinstance(self.transport, SSETransport):
            server_name = self.transport.url
            server_url = self.transport.url
            transport_type = self.transport.transport_type.value
        else:
            server_name = "Unknown MCP Server"
            server_url = None
            transport_type = (
                self.transport.transport_type.value
                if hasattr(self.transport, "transport_type")
                else None
            )

        return server_name, server_url, transport_type

    async def connect(self) -> Self:
        """Connect to MCP server and initialize session.

        Returns:
            Self for method chaining.

        Raises:
            ConnectionError: If connection fails.
            ImportError: If MCP SDK not available.
        """
        if self.connected:
            return self

        # Get server info for events
        server_name, server_url, transport_type = self._get_server_info()
        is_reconnect = self._was_connected

        # Emit connection started event
        started_at = datetime.now()
        crewai_event_bus.emit(
            self,
            MCPConnectionStartedEvent(
                server_name=server_name,
                server_url=server_url,
                transport_type=transport_type,
                is_reconnect=is_reconnect,
                connect_timeout=self.connect_timeout,
            ),
        )

        try:
            from mcp import ClientSession

            # Use AsyncExitStack to manage transport and session contexts together
            # This ensures they're in the same async scope and prevents cancel scope errors
            # Always enter transport context via exit stack (it handles already-connected state)
            await self._exit_stack.enter_async_context(self.transport)

            # Create ClientSession with transport streams
            self._session = ClientSession(
                self.transport.read_stream,
                self.transport.write_stream,
            )

            # Enter the session's async context manager via exit stack
            await self._exit_stack.enter_async_context(self._session)

            # Initialize the session (required by MCP protocol)
            try:
                await asyncio.wait_for(
                    self._session.initialize(),
                    timeout=self.connect_timeout,
                )
            except asyncio.CancelledError:
                # If initialization was cancelled (e.g., event loop closing),
                # cleanup and re-raise - don't suppress cancellation
                await self._cleanup_on_error()
                raise
            except BaseExceptionGroup as eg:
                # Handle exception groups from anyio task groups
                # Extract the actual meaningful error (not GeneratorExit)
                actual_error = None
                for exc in eg.exceptions:
                    if isinstance(exc, Exception) and not isinstance(
                        exc, GeneratorExit
                    ):
                        # Check if it's an HTTP error (like 401)
                        error_msg = str(exc).lower()
                        if "401" in error_msg or "unauthorized" in error_msg:
                            actual_error = exc
                            break
                        if "cancel scope" not in error_msg and "task" not in error_msg:
                            actual_error = exc
                            break

                await self._cleanup_on_error()
                if actual_error:
                    raise ConnectionError(
                        f"Failed to connect to MCP server: {actual_error}"
                    ) from actual_error
                raise ConnectionError(f"Failed to connect to MCP server: {eg}") from eg

            self._initialized = True
            self._was_connected = True

            completed_at = datetime.now()
            connection_duration_ms = (completed_at - started_at).total_seconds() * 1000
            crewai_event_bus.emit(
                self,
                MCPConnectionCompletedEvent(
                    server_name=server_name,
                    server_url=server_url,
                    transport_type=transport_type,
                    started_at=started_at,
                    completed_at=completed_at,
                    connection_duration_ms=connection_duration_ms,
                    is_reconnect=is_reconnect,
                ),
            )

            return self
        except ImportError as e:
            await self._cleanup_on_error()
            error_msg = (
                "MCP library not available. Please install with: pip install mcp"
            )
            self._emit_connection_failed(
                server_name,
                server_url,
                transport_type,
                error_msg,
                "import_error",
                started_at,
            )
            raise ImportError(error_msg) from e
        except asyncio.TimeoutError as e:
            await self._cleanup_on_error()
            error_msg = f"MCP connection timed out after {self.connect_timeout} seconds. The server may be slow or unreachable."
            self._emit_connection_failed(
                server_name,
                server_url,
                transport_type,
                error_msg,
                "timeout",
                started_at,
            )
            raise ConnectionError(error_msg) from e
        except asyncio.CancelledError:
            # Re-raise cancellation - don't suppress it
            await self._cleanup_on_error()
            self._emit_connection_failed(
                server_name,
                server_url,
                transport_type,
                "Connection cancelled",
                "cancelled",
                started_at,
            )
            raise
        except BaseExceptionGroup as eg:
            # Handle exception groups from anyio task groups at outer level
            actual_error = None
            for exc in eg.exceptions:
                if isinstance(exc, Exception) and not isinstance(exc, GeneratorExit):
                    error_msg = str(exc).lower()
                    if "401" in error_msg or "unauthorized" in error_msg:
                        actual_error = exc
                        break
                    if "cancel scope" not in error_msg and "task" not in error_msg:
                        actual_error = exc
                        break

            await self._cleanup_on_error()
            error_type = (
                "authentication"
                if actual_error
                and (
                    "401" in str(actual_error).lower()
                    or "unauthorized" in str(actual_error).lower()
                )
                else "network"
            )
            error_msg = str(actual_error) if actual_error else str(eg)
            self._emit_connection_failed(
                server_name,
                server_url,
                transport_type,
                error_msg,
                error_type,
                started_at,
            )
            if actual_error:
                raise ConnectionError(
                    f"Failed to connect to MCP server: {actual_error}"
                ) from actual_error
            raise ConnectionError(f"Failed to connect to MCP server: {eg}") from eg
        except Exception as e:
            await self._cleanup_on_error()
            error_type = (
                "authentication"
                if "401" in str(e).lower() or "unauthorized" in str(e).lower()
                else "network"
            )
            self._emit_connection_failed(
                server_name, server_url, transport_type, str(e), error_type, started_at
            )
            raise ConnectionError(f"Failed to connect to MCP server: {e}") from e

    def _emit_connection_failed(
        self,
        server_name: str,
        server_url: str | None,
        transport_type: str | None,
        error: str,
        error_type: str,
        started_at: datetime,
    ) -> None:
        """Emit connection failed event."""
        failed_at = datetime.now()
        crewai_event_bus.emit(
            self,
            MCPConnectionFailedEvent(
                server_name=server_name,
                server_url=server_url,
                transport_type=transport_type,
                error=error,
                error_type=error_type,
                started_at=started_at,
                failed_at=failed_at,
            ),
        )

    async def _cleanup_on_error(self) -> None:
        """Cleanup resources when an error occurs during connection."""
        try:
            await self._exit_stack.aclose()

        except Exception as e:
            # Best effort cleanup - ignore all other errors
            raise RuntimeError(f"Error during MCP client cleanup: {e}") from e
        finally:
            self._session = None
            self._initialized = False
            self._exit_stack = AsyncExitStack()

    async def disconnect(self) -> None:
        """Disconnect from MCP server and cleanup resources."""
        if not self.connected:
            return

        try:
            await self._exit_stack.aclose()
        except Exception as e:
            raise RuntimeError(f"Error during MCP client disconnect: {e}") from e
        finally:
            self._session = None
            self._initialized = False
            self._exit_stack = AsyncExitStack()

    async def list_tools(self, use_cache: bool | None = None) -> list[dict[str, Any]]:
        """List available tools from MCP server.

        Args:
            use_cache: Whether to use cached results. If None, uses
                client's cache_tools_list setting.

        Returns:
            List of tool definitions with name, description, and inputSchema.
        """
        if not self.connected:
            await self.connect()

        # Check cache if enabled
        use_cache = use_cache if use_cache is not None else self.cache_tools_list
        if use_cache:
            cache_key = self._get_cache_key("tools")
            if cache_key in _mcp_schema_cache:
                cached_data, cache_time = _mcp_schema_cache[cache_key]
                if time.time() - cache_time < _cache_ttl:
                    # Logger removed - return cached data
                    return cached_data

        # List tools with timeout and retries
        tools = await self._retry_operation(
            self._list_tools_impl,
            timeout=self.discovery_timeout,
        )

        # Cache results if enabled
        if use_cache:
            cache_key = self._get_cache_key("tools")
            _mcp_schema_cache[cache_key] = (tools, time.time())

        return tools

    async def _list_tools_impl(self) -> list[dict[str, Any]]:
        """Internal implementation of list_tools."""
        tools_result = await asyncio.wait_for(
            self.session.list_tools(),
            timeout=self.discovery_timeout,
        )

        return [
            {
                "name": sanitize_tool_name(tool.name),
                "description": getattr(tool, "description", ""),
                "inputSchema": getattr(tool, "inputSchema", {}),
            }
            for tool in tools_result.tools
        ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        if not self.connected:
            await self.connect()

        arguments = arguments or {}
        cleaned_arguments = self._clean_tool_arguments(arguments)

        # Get server info for events
        server_name, server_url, transport_type = self._get_server_info()

        # Emit tool execution started event
        started_at = datetime.now()
        crewai_event_bus.emit(
            self,
            MCPToolExecutionStartedEvent(
                server_name=server_name,
                server_url=server_url,
                transport_type=transport_type,
                tool_name=tool_name,
                tool_args=cleaned_arguments,
            ),
        )

        try:
            result = await self._retry_operation(
                lambda: self._call_tool_impl(tool_name, cleaned_arguments),
                timeout=self.execution_timeout,
            )

            completed_at = datetime.now()
            execution_duration_ms = (completed_at - started_at).total_seconds() * 1000
            crewai_event_bus.emit(
                self,
                MCPToolExecutionCompletedEvent(
                    server_name=server_name,
                    server_url=server_url,
                    transport_type=transport_type,
                    tool_name=tool_name,
                    tool_args=cleaned_arguments,
                    result=result,
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_duration_ms=execution_duration_ms,
                ),
            )

            return result
        except Exception as e:
            failed_at = datetime.now()
            error_type = (
                "timeout"
                if isinstance(e, (asyncio.TimeoutError, ConnectionError))
                and "timeout" in str(e).lower()
                else "server_error"
            )
            crewai_event_bus.emit(
                self,
                MCPToolExecutionFailedEvent(
                    server_name=server_name,
                    server_url=server_url,
                    transport_type=transport_type,
                    tool_name=tool_name,
                    tool_args=cleaned_arguments,
                    error=str(e),
                    error_type=error_type,
                    started_at=started_at,
                    failed_at=failed_at,
                ),
            )
            raise

    def _clean_tool_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Clean tool arguments by removing None values and fixing formats.

        Args:
            arguments: Raw tool arguments.

        Returns:
            Cleaned arguments ready for MCP server.
        """
        cleaned = {}

        for key, value in arguments.items():
            # Skip None values
            if value is None:
                continue

            # Fix sources array format: convert ["web"] to [{"type": "web"}]
            if key == "sources" and isinstance(value, list):
                fixed_sources = []
                for item in value:
                    if isinstance(item, str):
                        # Convert string to object format
                        fixed_sources.append({"type": item})
                    elif isinstance(item, dict):
                        # Already in correct format
                        fixed_sources.append(item)
                    else:
                        # Keep as is if unknown format
                        fixed_sources.append(item)
                if fixed_sources:
                    cleaned[key] = fixed_sources
                continue

            # Recursively clean nested dictionaries
            if isinstance(value, dict):
                nested_cleaned = self._clean_tool_arguments(value)
                if nested_cleaned:  # Only add if not empty
                    cleaned[key] = nested_cleaned
            elif isinstance(value, list):
                # Clean list items
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = self._clean_tool_arguments(item)
                        if cleaned_item:
                            cleaned_list.append(cleaned_item)
                    elif item is not None:
                        cleaned_list.append(item)
                if cleaned_list:
                    cleaned[key] = cleaned_list
            else:
                # Keep primitive values
                cleaned[key] = value

        return cleaned

    async def _call_tool_impl(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Internal implementation of call_tool."""
        result = await asyncio.wait_for(
            self.session.call_tool(tool_name, arguments),
            timeout=self.execution_timeout,
        )

        # Extract result content
        if hasattr(result, "content") and result.content:
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    return str(content_item.text)
                return str(content_item)
            return str(result.content)

        return str(result)

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts from MCP server.

        Returns:
            List of prompt definitions.
        """
        if not self.connected:
            await self.connect()

        return await self._retry_operation(
            self._list_prompts_impl,
            timeout=self.discovery_timeout,
        )

    async def _list_prompts_impl(self) -> list[dict[str, Any]]:
        """Internal implementation of list_prompts."""
        prompts_result = await asyncio.wait_for(
            self.session.list_prompts(),
            timeout=self.discovery_timeout,
        )

        return [
            {
                "name": prompt.name,
                "description": getattr(prompt, "description", ""),
                "arguments": getattr(prompt, "arguments", []),
            }
            for prompt in prompts_result.prompts
        ]

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get a prompt from the MCP server.

        Args:
            prompt_name: Name of the prompt to get.
            arguments: Optional prompt arguments.

        Returns:
            Prompt content and metadata.
        """
        if not self.connected:
            await self.connect()

        arguments = arguments or {}

        return await self._retry_operation(
            lambda: self._get_prompt_impl(prompt_name, arguments),
            timeout=self.execution_timeout,
        )

    async def _get_prompt_impl(
        self, prompt_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Internal implementation of get_prompt."""
        result = await asyncio.wait_for(
            self.session.get_prompt(prompt_name, arguments),
            timeout=self.execution_timeout,
        )

        return {
            "name": prompt_name,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in result.messages
            ],
            "arguments": arguments,
        }

    async def _retry_operation(
        self,
        operation: Callable[[], Any],
        timeout: int | None = None,
    ) -> Any:
        """Retry an operation with exponential backoff.

        Args:
            operation: Async operation to retry.
            timeout: Operation timeout in seconds.

        Returns:
            Operation result.
        """
        last_error = None
        timeout = timeout or self.execution_timeout

        for attempt in range(self.max_retries):
            try:
                if timeout:
                    return await asyncio.wait_for(operation(), timeout=timeout)
                return await operation()

            except asyncio.TimeoutError as e:  # noqa: PERF203
                last_error = f"Operation timed out after {timeout} seconds"
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise ConnectionError(last_error) from e

            except Exception as e:
                error_str = str(e).lower()

                # Classify errors as retryable or non-retryable
                if "authentication" in error_str or "unauthorized" in error_str:
                    raise ConnectionError(f"Authentication failed: {e}") from e

                if "not found" in error_str:
                    raise ValueError(f"Resource not found: {e}") from e

                # Retryable errors
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise ConnectionError(
                        f"Operation failed after {self.max_retries} attempts: {last_error}"
                    ) from e

        raise ConnectionError(f"Operation failed: {last_error}")

    def _get_cache_key(self, resource_type: str) -> str:
        """Generate cache key for resource.

        Args:
            resource_type: Type of resource (e.g., "tools", "prompts").

        Returns:
            Cache key string.
        """
        # Use transport type and URL/command as cache key
        if isinstance(self.transport, StdioTransport):
            key = f"stdio:{self.transport.command}:{':'.join(self.transport.args)}"
        elif isinstance(self.transport, HTTPTransport):
            key = f"http:{self.transport.url}"
        elif isinstance(self.transport, SSETransport):
            key = f"sse:{self.transport.url}"
        else:
            key = f"{self.transport.transport_type}:unknown"

        return f"mcp:{key}:{resource_type}"

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
