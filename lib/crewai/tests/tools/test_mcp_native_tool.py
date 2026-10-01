"""Tests for MCPNativeTool's synchronous bridge over an async MCP client."""

import asyncio
from typing import Any

import pytest

from crewai.mcp.client import _MCPToolResult
from crewai.tools.mcp_native_tool import MCPNativeTool


class DisconnectFailsClient:
    """Mimics MCPClient: the tool call answers, disconnect() fails.

    MCPClient.disconnect() wraps any non-MCPConnectionError teardown failure
    in RuntimeError("Error during MCP client disconnect: ...").
    """

    async def connect(self) -> Any:
        return self

    async def call_tool_result(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> _MCPToolResult:
        return _MCPToolResult("tool result", False)

    async def disconnect(self) -> None:
        raise RuntimeError("Error during MCP client disconnect: boom")


def _tool() -> MCPNativeTool:
    return MCPNativeTool(
        client_factory=DisconnectFailsClient,
        tool_name="post",
        tool_schema={"description": "post a message"},
        server_name="slack",
    )


class TestDisconnectFailureSurfaces:
    """A RuntimeError raised while the tool runs must reach the caller.

    ``_run`` detects "no running event loop" via ``get_running_loop()``'s
    RuntimeError; if that check also swallows a RuntimeError from the
    guarded block, the in-loop fallback ``asyncio.run()`` replaces the real
    failure with "cannot be called from a running event loop".
    """

    def test_inside_running_loop_reports_the_real_error(self) -> None:
        async def call_tool() -> Any:
            return _tool().run(query="hi")

        with pytest.raises(RuntimeError, match="Error during MCP client disconnect"):
            asyncio.run(call_tool())

    def test_outside_running_loop_reports_the_real_error(self) -> None:
        with pytest.raises(RuntimeError, match="Error during MCP client disconnect"):
            _tool().run(query="hi")
