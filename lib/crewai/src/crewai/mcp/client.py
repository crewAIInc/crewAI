"""MCP client with session management for CrewAI agents."""

import asyncio
from collections.abc import Callable, Coroutine
from contextlib import AsyncExitStack
from datetime import datetime
import logging
import sys
import time
from typing import Any, NamedTuple, TypeVar
from urllib.parse import urlparse

from typing_extensions import Self


if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.mcp_events import (
    MCPConnectionCompletedEvent,
    MCPConnectionFailedEvent,
    MCPConnectionStartedEvent,
    MCPToolExecutionCompletedEvent,
    MCPToolExecutionFailedEvent,
    MCPToolExecutionStartedEvent,
)
from crewai.mcp.exceptions import (
    MCPConnectionError,
    error_for_status,
    error_type_for_status,
    find_http_status,
    find_transport_failure,
    tool_execution_error_type,
)
from crewai.mcp.transports.base import BaseTransport
from crewai.mcp.transports.http import HTTPTransport
from crewai.mcp.transports.sse import SSETransport
from crewai.mcp.transports.stdio import StdioTransport
from crewai.utilities.string_utils import sanitize_tool_name


class _MCPToolResult(NamedTuple):
    """Internal result from an MCP tool call, carrying the ``isError`` flag."""

    content: str
    is_error: bool


MCP_CONNECTION_TIMEOUT = 30  # Increased for slow servers
MCP_TOOL_EXECUTION_TIMEOUT = 30
MCP_DISCOVERY_TIMEOUT = 30  # Increased for slow servers
MCP_MAX_RETRIES = 3

_T = TypeVar("_T")

_mcp_schema_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_cache_ttl = 300  # 5 minutes


def _server_name_from_url(url: str) -> str:
    return urlparse(url).hostname or url


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
            server_url = self.transport.url
            server_name = _server_name_from_url(server_url)
            transport_type = self.transport.transport_type.value
        elif isinstance(self.transport, SSETransport):
            server_url = self.transport.url
            server_name = _server_name_from_url(server_url)
            transport_type = self.transport.transport_type.value
        else:
            server_name = "Unknown MCP Server"
            server_url = None
            transport_type = self.transport.transport_type.value

        return server_name, server_url, transport_type

    async def connect(self) -> Self:
        """Connect to MCP server and initialize session.

        Returns:
            Self for method chaining.

        Raises:
            MCPAuthenticationError: If the server refused the connection with
                an authentication status.
            MCPHTTPError: If the server refused the connection with any other
                HTTP status.
            MCPConnectionError: If the connection failed for any other reason.
            ImportError: If MCP SDK not available.
        """
        if self.connected:
            return self

        server_info = self._get_server_info()
        server_name, server_url, transport_type = server_info
        is_reconnect = self._was_connected

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

            self._session = ClientSession(
                self.transport.read_stream,
                self.transport.write_stream,
            )

            await self._exit_stack.enter_async_context(self._session)

            # MCP protocol requires session.initialize() before any other request.
            # Failures propagate to the handlers below, which unwind the transport
            # once and inspect what that unwinding reveals about the cause.
            await asyncio.wait_for(
                self._session.initialize(),
                timeout=self.connect_timeout,
            )

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
            cleanup_error = await self._cleanup_on_error()
            status_code = find_http_status(e, cleanup_error)
            if status_code is not None:
                raise self._report_connection_failure(
                    server_info, started_at, status_code=status_code
                ) from e
            error_msg = f"MCP connection timed out after {self.connect_timeout} seconds. The server may be slow or unreachable."
            self._emit_connection_failed(
                server_name,
                server_url,
                transport_type,
                error_msg,
                "timeout",
                started_at,
            )
            raise MCPConnectionError(error_msg) from e
        except asyncio.CancelledError as e:
            # A failing transport cancels this coroutine, so cancellation alone
            # says nothing. Unwinding the transport is what reveals the cause.
            cleanup_error = await self._cleanup_on_error()
            status_code = find_http_status(e, cleanup_error)
            if status_code is not None:
                raise self._report_connection_failure(
                    server_info, started_at, status_code=status_code
                ) from e
            if (failure := find_transport_failure(cleanup_error)) is not None:
                raise self._report_connection_failure(
                    server_info, started_at, error=failure
                ) from failure
            # Nothing failed, so this is a real cancellation: never swallow it.
            self._emit_connection_failed(
                server_name,
                server_url,
                transport_type,
                "Connection cancelled",
                "cancelled",
                started_at,
            )
            raise
        except (BaseExceptionGroup, Exception) as e:
            cleanup_error = await self._cleanup_on_error()
            status_code = find_http_status(e, cleanup_error)
            if status_code is not None:
                raise self._report_connection_failure(
                    server_info, started_at, status_code=status_code
                ) from e
            failure = find_transport_failure(e, cleanup_error) or e
            raise self._report_connection_failure(
                server_info, started_at, error=failure
            ) from e

    def _report_connection_failure(
        self,
        server_info: tuple[str, str | None, str | None],
        started_at: datetime,
        *,
        error: BaseException | None = None,
        status_code: int | None = None,
    ) -> MCPConnectionError:
        """Build a connection failure, emit the event, and return it to raise."""
        if status_code is not None:
            failure = error_for_status(status_code)
            error_msg = str(failure)
            error_type = error_type_for_status(status_code) or "network"
        elif error is not None and isinstance(error, MCPConnectionError):
            failure = error
            error_msg = str(error)
            error_type = (
                error_type_for_status(error.status_code) or "network"
                if error.status_code is not None
                else "network"
            )
            status_code = error.status_code
        elif error is not None:
            error_msg = f"Failed to connect to MCP server: {error}"
            error_type = "network"
            status_code = find_http_status(error)
            failure = MCPConnectionError(error_msg, status_code=status_code)
        else:
            raise ValueError("Either error or status_code must be provided")

        server_name, server_url, transport_type = server_info
        self._emit_connection_failed(
            server_name,
            server_url,
            transport_type,
            error_msg,
            error_type,
            started_at,
            status_code=status_code,
        )
        return failure

    def _emit_connection_failed(
        self,
        server_name: str,
        server_url: str | None,
        transport_type: str | None,
        error: str,
        error_type: str,
        started_at: datetime,
        status_code: int | None = None,
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
                status_code=status_code,
                started_at=started_at,
                failed_at=failed_at,
            ),
        )

    async def _cleanup_on_error(self) -> BaseException | None:
        """Cleanup resources when an error occurs during connection.

        Returns:
            The exception raised while unwinding the transport, if any. The
            transport reports the server's refusal here rather than to the
            coroutine that was waiting, so the caller inspects this for an
            HTTP status instead of treating it as a cleanup problem.
        """
        try:
            await self._exit_stack.aclose()
        except asyncio.CancelledError as e:
            return e
        except (Exception, BaseExceptionGroup) as e:
            # Groups are caught explicitly because one holding only BaseExceptions
            # is not an Exception, yet can still carry the server's refusal.
            return e
        else:
            return None
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
        except asyncio.CancelledError:
            raise
        except MCPConnectionError:
            raise
        except (Exception, BaseExceptionGroup) as e:
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

        use_cache = use_cache if use_cache is not None else self.cache_tools_list
        if use_cache:
            cache_key = self._get_cache_key("tools")
            if cache_key in _mcp_schema_cache:
                cached_data, cache_time = _mcp_schema_cache[cache_key]
                if time.time() - cache_time < _cache_ttl:
                    return cached_data

        tools = await self._retry_operation(
            self._list_tools_impl,
            timeout=self.discovery_timeout,
        )

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
                "original_name": tool.name,
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
            Tool execution result content. The ``isError`` flag is dropped;
            use :meth:`call_tool_result` when the caller needs it.
        """
        return (await self.call_tool_result(tool_name, arguments)).content

    async def call_tool_result(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> _MCPToolResult:
        """Call a tool and return its content together with the ``isError`` flag.

        MCP servers report a failed tool as a *successful* JSON-RPC response
        carrying ``isError: true``. Callers that only take the content cannot
        tell that apart from a normal result, which is how a failed step ends
        up looking like a successful one.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            The content string plus whether the server flagged it as an error.
        """
        if not self.connected:
            await self.connect()

        arguments = arguments or {}
        cleaned_arguments = self._clean_tool_arguments(arguments)

        server_name, server_url, transport_type = self._get_server_info()

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
            tool_result: _MCPToolResult = await self._retry_operation(
                lambda: self._call_tool_impl(tool_name, cleaned_arguments),
                timeout=self.execution_timeout,
            )

            finished_at = datetime.now()
            execution_duration_ms = (finished_at - started_at).total_seconds() * 1000

            if tool_result.is_error:
                crewai_event_bus.emit(
                    self,
                    MCPToolExecutionFailedEvent(
                        server_name=server_name,
                        server_url=server_url,
                        transport_type=transport_type,
                        tool_name=tool_name,
                        tool_args=cleaned_arguments,
                        error=tool_result.content,
                        error_type="tool_error",
                        started_at=started_at,
                        failed_at=finished_at,
                    ),
                )
            else:
                crewai_event_bus.emit(
                    self,
                    MCPToolExecutionCompletedEvent(
                        server_name=server_name,
                        server_url=server_url,
                        transport_type=transport_type,
                        tool_name=tool_name,
                        tool_args=cleaned_arguments,
                        result=tool_result.content,
                        started_at=started_at,
                        completed_at=finished_at,
                        execution_duration_ms=execution_duration_ms,
                    ),
                )

            return tool_result
        except Exception as e:
            failed_at = datetime.now()
            crewai_event_bus.emit(
                self,
                MCPToolExecutionFailedEvent(
                    server_name=server_name,
                    server_url=server_url,
                    transport_type=transport_type,
                    tool_name=tool_name,
                    tool_args=cleaned_arguments,
                    error=str(e),
                    error_type=tool_execution_error_type(e),
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
        cleaned: dict[str, Any] = {}

        for key, value in arguments.items():
            if value is None:
                continue

            # Normalize sources from ["web"] to [{"type": "web"}]
            if key == "sources" and isinstance(value, list):
                fixed_sources = []
                for item in value:
                    if isinstance(item, str):
                        fixed_sources.append({"type": item})
                    elif isinstance(item, dict):
                        fixed_sources.append(item)
                    else:
                        fixed_sources.append(item)
                if fixed_sources:
                    cleaned[key] = fixed_sources
                continue

            if isinstance(value, dict):
                nested_cleaned = self._clean_tool_arguments(value)
                if nested_cleaned:  # Only add if not empty
                    cleaned[key] = nested_cleaned
            elif isinstance(value, list):
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
                cleaned[key] = value

        return cleaned

    async def _call_tool_impl(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> _MCPToolResult:
        """Internal implementation of call_tool."""
        result = await asyncio.wait_for(
            self.session.call_tool(tool_name, arguments),
            timeout=self.execution_timeout,
        )

        is_error = getattr(result, "isError", False) or False

        if hasattr(result, "content") and result.content:
            if isinstance(result.content, list) and len(result.content) > 0:
                content_item = result.content[0]
                if hasattr(content_item, "text"):
                    return _MCPToolResult(str(content_item.text), is_error)
                return _MCPToolResult(str(content_item), is_error)
            return _MCPToolResult(str(result.content), is_error)

        return _MCPToolResult(str(result), is_error)

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
        operation: Callable[[], Coroutine[Any, Any, _T]],
        timeout: int | None = None,
    ) -> _T:
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
                if isinstance(e, MCPConnectionError):
                    raise

                status_code = find_http_status(e)
                if status_code is not None and (
                    error_type_for_status(status_code) == "authentication"
                ):
                    raise error_for_status(status_code, detail=str(e)) from e

                error_str = str(e).lower()

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
