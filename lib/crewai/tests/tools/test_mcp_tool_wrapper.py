"""Tests for MCPToolWrapper, the external-URL MCP resolution path."""

from contextlib import asynccontextmanager
from unittest.mock import patch

import mcp.types as mcp_types

from crewai.tools.mcp_tool_wrapper import MCPToolWrapper
from crewai.tools.tool_failure import ToolFailure, ToolFailureReason


def _call_tool_result(*, is_error: bool) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Error: file not found")],
        isError=is_error,
    )


class _FakeSession:
    def __init__(self, result: mcp_types.CallToolResult):
        self._result = result

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def initialize(self) -> None:
        return None

    async def call_tool(self, _name: str, _arguments: dict[str, object]) -> object:
        return self._result


@asynccontextmanager
async def _fake_streamablehttp(*_args: object, **_kwargs: object):
    yield (None, None, None)


class TestMcpToolWrapperIsError:
    """An MCP server flags a failed call with isError on a 200 response."""

    @staticmethod
    def _run(result: mcp_types.CallToolResult) -> object:
        tool = MCPToolWrapper(
            mcp_server_params={"url": "https://mcp.example.com/mcp"},
            tool_name="read_file",
            tool_schema={"description": "Read a file"},
            server_name="mcp_example_com",
        )
        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                _fake_streamablehttp,
            ),
            patch("mcp.ClientSession", lambda *_a, **_kw: _FakeSession(result)),
        ):
            return tool.run(path="/etc/hosts")

    def test_is_error_becomes_a_tool_failure(self) -> None:
        result = self._run(_call_tool_result(is_error=True))

        assert isinstance(result, ToolFailure)
        assert result.reason is ToolFailureReason.MCP_ERROR
        assert result.message == "Error: file not found"
        assert result.details["server"] == "mcp_example_com"
        assert result.details["tool"] == "read_file"

    def test_successful_call_still_returns_plain_text(self) -> None:
        assert self._run(_call_tool_result(is_error=False)) == "Error: file not found"
