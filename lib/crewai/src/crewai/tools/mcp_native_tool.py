"""Native MCP tool wrapper for CrewAI agents.

This module provides a tool wrapper that creates a fresh MCP client for every
invocation, ensuring safe parallel execution even when the same tool is called
concurrently by the executor.
"""

import asyncio
from collections.abc import Callable
import contextvars
import re
from typing import Any

from crewai.tools import BaseTool


_URL_PATTERN = re.compile(r"[a-zA-Z]+://[^\s\"']+")


class MCPNativeTool(BaseTool):
    """Native MCP tool that creates a fresh client per invocation.

    A ``client_factory`` callable produces an independent ``MCPClient`` +
    transport for every ``_run_async`` call.  This guarantees that parallel
    invocations -- whether of the *same* tool or *different* tools from the
    same server -- never share mutable connection state (which would cause
    anyio cancel-scope errors).
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        tool_name: str,
        tool_schema: dict[str, Any],
        server_name: str,
        original_tool_name: str | None = None,
    ) -> None:
        """Initialize native MCP tool.

        Args:
            client_factory: Zero-arg callable that returns a new MCPClient.
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

        self._client_factory = client_factory
        self._original_tool_name = original_tool_name or tool_name
        self._server_name = server_name

    @property
    def original_tool_name(self) -> str:
        """Get the original tool name."""
        return self._original_tool_name

    @property
    def server_name(self) -> str:
        """Get the server name."""
        return self._server_name

    def _run(self, **kwargs: Any) -> str:
        """Execute tool using the MCP client session.

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            Result from the MCP tool execution.
        """
        try:
            _validate_mcp_tool_args_for_urls(kwargs)

            try:
                asyncio.get_running_loop()

                import concurrent.futures

                ctx = contextvars.copy_context()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    coro = self._run_async(**kwargs)
                    future = executor.submit(ctx.run, asyncio.run, coro)
                    return future.result()
            except RuntimeError:
                return asyncio.run(self._run_async(**kwargs))

        except ValueError as e:
            return f"SSRF protection blocked MCP tool execution: {e}"
        except Exception as e:
            raise RuntimeError(
                f"Error executing MCP tool {self.original_tool_name}: {e!s}"
            ) from e


def _validate_mcp_tool_args_for_urls(kwargs: dict[str, Any]) -> None:
    """Scan MCP tool arguments for URLs and validate them against SSRF rules.

    Recursively scans string values for http/https URLs and validates each
    one. This prevents agents from using MCP tools to access internal
    services, cloud metadata endpoints, or other private resources.

    Raises:
        ValueError: If any URL in the arguments resolves to a private/reserved IP.
    """
    from crewai_tools.security.safe_path import validate_url_and_resolve

    for value in kwargs.values():
        if isinstance(value, str):
            for match in _URL_PATTERN.finditer(value):
                validate_url_and_resolve(match.group())
        elif isinstance(value, dict):
            _validate_mcp_tool_args_for_urls(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    for match in _URL_PATTERN.finditer(item):
                        validate_url_and_resolve(match.group())
                elif isinstance(item, dict):
                    _validate_mcp_tool_args_for_urls(item)

    async def _run_async(self, **kwargs: Any) -> str:
        """Async implementation of tool execution.

        A fresh ``MCPClient`` is created for every invocation so that
        concurrent calls never share transport or session state.

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            Result from the MCP tool execution.
        """
        client = self._client_factory()
        await client.connect()

        try:
            result = await client.call_tool(self.original_tool_name, kwargs)
        finally:
            await client.disconnect()

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
