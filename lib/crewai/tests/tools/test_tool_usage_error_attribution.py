"""`Tool Usage Error` spans must carry the tool name.

The emitter has accepted `tool_name` since it was written
(`Telemetry.tool_usage_error`, which writes the attribute only `if tool_name:`), but no caller
passed it, so tool errors were attributed to the empty string and no per-tool error rate could
be computed at all. The omission is silent by construction - a missing kwarg simply produces a
span without the attribute - so it needs tests rather than review attention.

Both error paths exist twice, once in the sync `_use` and once in the async `_ause`, so the
matrix here is (usage-limit, execution-error) x (sync, async). The parsing failure in
`_tool_calling` is covered too, pinning the opposite expectation: it must stay unattributed.
"""

from unittest.mock import MagicMock

import pytest

from crewai.tools import BaseTool
from crewai.tools.tool_usage import ToolUsage
from crewai.tools.tool_calling import ToolCalling
from crewai.utilities.string_utils import sanitize_tool_name


class ExplodingTool(BaseTool):
    name: str = "Exploding Tool"
    description: str = "Raises on every call, to exercise the execution-error path"

    def _run(self, text: str) -> str:
        raise RuntimeError("boom")


class LimitedTool(BaseTool):
    name: str = "Limited Tool"
    description: str = "Has a usage limit, to exercise the usage-limit path"
    max_usage_count: int = 1

    def _run(self, text: str) -> str:
        return f"ok {text}"


def build_usage(tool: BaseTool) -> ToolUsage:
    """A ToolUsage wired to one tool, with a mocked telemetry sink."""
    # Real strings on the mocks: ToolUsageStartedEvent is a pydantic model and rejects
    # MagicMock attributes, so a bare MagicMock() fails validation before the code under
    # test is reached.
    task = MagicMock()
    task.name = "some task"
    agent = MagicMock()
    agent.role = "some role"
    agent.key = "agent-key"
    agent._original_role = "some role"
    agent.verbose = False
    action = MagicMock()
    action.tool = tool.name
    action.tool_input = {"text": "x"}

    usage = ToolUsage(
        tools_handler=MagicMock(cache=None, last_used_tool=None),
        tools=[tool],
        task=task,
        function_calling_llm=None,
        agent=agent,
        action=action,
    )
    usage._telemetry = MagicMock()
    # Report the error on the first failure instead of after the retry budget.
    usage._max_parsing_attempts = 0
    return usage


def recorded_tool_name(usage: ToolUsage) -> str | None:
    """The tool_name the code under test handed to telemetry."""
    assert usage._telemetry.tool_usage_error.called, (
        "expected a Tool Usage Error span to be emitted"
    )
    return usage._telemetry.tool_usage_error.call_args.kwargs.get("tool_name")


def calling_for(tool: BaseTool) -> ToolCalling:
    return ToolCalling(tool_name=tool.name, arguments={"text": "x"})


def test_sync_execution_error_reports_the_tool_name():
    tool = ExplodingTool()
    usage = build_usage(tool)

    usage.use(calling=calling_for(tool), tool_string="Action: Exploding Tool")

    recorded = recorded_tool_name(usage)
    assert recorded == sanitize_tool_name(tool.name)
    assert recorded, "an empty tool_name reproduces the bug: the emitter skips falsy values"


@pytest.mark.asyncio
async def test_async_execution_error_reports_the_tool_name():
    tool = ExplodingTool()
    usage = build_usage(tool)

    await usage.ause(calling=calling_for(tool), tool_string="Action: Exploding Tool")

    recorded = recorded_tool_name(usage)
    assert recorded == sanitize_tool_name(tool.name)
    assert recorded, "an empty tool_name reproduces the bug: the emitter skips falsy values"


def test_sync_usage_limit_error_reports_the_tool_name():
    tool = LimitedTool()
    tool.current_usage_count = tool.max_usage_count
    usage = build_usage(tool)

    usage.use(calling=calling_for(tool), tool_string="Action: Limited Tool")

    recorded = recorded_tool_name(usage)
    assert recorded == sanitize_tool_name(tool.name)
    assert recorded, "an empty tool_name reproduces the bug: the emitter skips falsy values"


@pytest.mark.asyncio
async def test_async_usage_limit_error_reports_the_tool_name():
    tool = LimitedTool()
    tool.current_usage_count = tool.max_usage_count
    usage = build_usage(tool)

    await usage.ause(calling=calling_for(tool), tool_string="Action: Limited Tool")

    recorded = recorded_tool_name(usage)
    assert recorded == sanitize_tool_name(tool.name)
    assert recorded, "an empty tool_name reproduces the bug: the emitter skips falsy values"


def test_parsing_failure_stays_unattributed():
    """A tool-call parsing failure has no tool identity, and must not invent one.

    Passing the raw tool_string here would put unbounded model output into a metrics
    dimension, which is worse than an honest empty name.
    """
    usage = build_usage(ExplodingTool())
    # Force both parse attempts to raise. Without this the fallback returns a ToolUsageError
    # object instead of raising, so the telemetry branch is never reached and the test would
    # pass for the wrong reason.
    usage._original_tool_calling = MagicMock(side_effect=RuntimeError("unparseable"))

    usage._tool_calling("not a parseable tool call at all")

    assert recorded_tool_name(usage) is None
