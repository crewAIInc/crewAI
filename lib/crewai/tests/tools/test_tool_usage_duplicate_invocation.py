"""Regression tests for retrying a failing tool inside one ToolUsage attempt."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crewai.tools import BaseTool
from crewai.tools.tool_calling import ToolCalling
from crewai.tools.tool_usage import ToolUsage


class ExplodingTool(BaseTool):
    """Tool that fails every time and records each execution."""

    name: str = "exploding_tool"
    description: str = "Always raises so attempts can be counted"
    calls: list[str] = []

    def _run(self, value: str) -> str:
        type(self).calls.append(value)
        raise RuntimeError("upstream failure")


def _tool_usage() -> ToolUsage:
    return ToolUsage(
        tools_handler=None,
        tools=[ExplodingTool().to_structured_tool()],
        task=None,
        function_calling_llm=MagicMock(),
        agent=None,
        action=MagicMock(),
    )


def _calling() -> ToolCalling:
    return ToolCalling(tool_name="exploding_tool", arguments={"value": "x"})


def test_failing_tool_runs_once_per_sync_attempt() -> None:
    ExplodingTool.calls = []
    usage = _tool_usage()

    usage.use(
        calling=_calling(),
        tool_string='Action: exploding_tool\nAction Input: {"value": "x"}',
    )

    assert len(ExplodingTool.calls) == usage._max_parsing_attempts


@pytest.mark.asyncio
async def test_failing_tool_runs_once_per_async_attempt() -> None:
    ExplodingTool.calls = []
    usage = _tool_usage()

    await usage.ause(
        calling=_calling(),
        tool_string='Action: exploding_tool\nAction Input: {"value": "x"}',
    )

    assert len(ExplodingTool.calls) == usage._max_parsing_attempts

