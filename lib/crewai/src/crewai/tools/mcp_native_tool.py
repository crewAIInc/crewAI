"""Native MCP tool wrapper for CrewAI agents.

This module provides a tool wrapper that creates a fresh MCP client for every
invocation, ensuring safe parallel execution even when the same tool is called
concurrently by the executor.
"""

import asyncio
from collections.abc import Callable
import contextvars
from typing import Any

from crewai.tools import BaseTool
from crewai.tools.tool_failure import ToolFailure, ToolFailureReason


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

    def _run(self, **kwargs: Any) -> Any:
        """Execute tool using the MCP client session.

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            The tool's text result, or a :class:`ToolFailure` when the server
            answered with ``isError: true``.
        """
        try:
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

        except Exception as e:
            raise RuntimeError(
                f"Error executing MCP tool {self.original_tool_name}: {e!s}"
            ) from e

    async def _run_async(self, **kwargs: Any) -> Any:
        """Async implementation of tool execution.

        A fresh ``MCPClient`` is created for every invocation so that
        concurrent calls never share transport or session state.

        Args:
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            The tool's text result, or a :class:`ToolFailure` when the server
            answered with ``isError: true``.
        """
        client = self._client_factory()
        await client.connect()

        try:
            tool_result = await client.call_tool_result(self.original_tool_name, kwargs)
        finally:
            await client.disconnect()

        content = self._extract_content(tool_result.content)

        if tool_result.is_error:
            # isError rides on an otherwise successful response; without this
            # the agent cannot tell it apart from a real result.
            return ToolFailure(
                message=content,
                reason=ToolFailureReason.MCP_ERROR,
                details={
                    "server": self._server_name,
                    "tool": self._original_tool_name,
                },
            )

        return content

    @staticmethod
    def _extract_content(result: Any) -> str:
        """Flatten an MCP result payload into the text the agent sees."""
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
