from unittest.mock import AsyncMock

import pytest

from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


def _wrapper() -> MCPToolWrapper:
    return MCPToolWrapper(
        mcp_server_params={"url": "https://example.test/mcp"},
        tool_name="echo",
        tool_schema={"description": "echo"},
        server_name="demo",
    )


def test_run_without_running_loop():
    wrapper = _wrapper()
    wrapper._run_async = AsyncMock(return_value="ok")
    assert wrapper._run(text="hi") == "ok"


@pytest.mark.asyncio
async def test_run_from_running_loop():
    wrapper = _wrapper()
    wrapper._run_async = AsyncMock(return_value="ok")
    # Call on this thread so get_running_loop() sees the pytest loop and
    # takes the worker-thread path instead of asyncio.run() in-process.
    result = wrapper._run(text="hi")
    assert result == "ok"
    assert "cannot be called from a running event loop" not in result
