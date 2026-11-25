"""The sync bridge of MCPNativeTool must report the tool's own error.

``_run`` used to wrap both the running-loop probe and the threaded execution
in a single ``except RuntimeError``, so a ``RuntimeError`` raised by the tool
was mistaken for the "no running loop" signal: the fallback called
``asyncio.run`` again, replaced the real message with
"asyncio.run() cannot be called from a running event loop" and ran the tool a
second time.
"""

import asyncio

import pytest

from crewai.tools.mcp_native_tool import MCPNativeTool


def _tool(calls: list[int]) -> MCPNativeTool:
    class _FailingTool(MCPNativeTool):
        async def _run_async(self, **kwargs: object) -> object:
            calls.append(1)
            raise RuntimeError("server said no")

    return _FailingTool(
        client_factory=lambda: None,
        tool_name="echo",
        tool_schema={},
        server_name="srv",
    )


def test_keeps_the_tool_error_inside_a_running_loop() -> None:
    calls: list[int] = []
    tool = _tool(calls)

    async def main() -> None:
        with pytest.raises(RuntimeError, match="server said no"):
            tool._run()

    asyncio.run(main())
    assert len(calls) == 1, "the tool must not be run twice"


def test_keeps_the_tool_error_without_a_running_loop() -> None:
    calls: list[int] = []
    tool = _tool(calls)

    with pytest.raises(RuntimeError, match="server said no"):
        tool._run()

    assert len(calls) == 1


def test_still_returns_the_result_inside_a_running_loop() -> None:
    class _EchoTool(MCPNativeTool):
        async def _run_async(self, **kwargs: object) -> str:
            return "pong"

    tool = _EchoTool(
        client_factory=lambda: None,
        tool_name="echo",
        tool_schema={},
        server_name="srv",
    )

    async def main() -> None:
        assert tool._run() == "pong"

    asyncio.run(main())
