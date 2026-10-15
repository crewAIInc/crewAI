"""A native function-calling tool call served from the cache says so on its event.

Every native tool path reads the tools handler's cache and skips the tool body
on a hit, but a `ToolUsageFinishedEvent` emitted without `from_cache` makes every
consumer of the bus (tracing included) see a replayed call as a live one. The
text-protocol path (`ToolUsage`) has always carried the flag.

Three paths compute the flag and all three must report it:

- `CrewAgentExecutor._execute_single_native_tool_call` (deprecated executor),
- `AgentExecutor._execute_single_native_tool_call` (the default executor),
- `agent_utils.execute_single_native_tool_call`, which `StepExecutor` uses for
  the default executor's plan-and-execute steps.

All three read the cache *before* the `before_tool_call` hooks run, so a hook
that blocks the call replaces the cached result with a blocked message. The flag
has to be cleared there for the same reason the cached `ToolFailure` already is:
nothing the caller receives came from the cache, and the consumers read the flag
as a statement about the result they were handed -- the tracing handler labels
the span's `tool_call_result`, and the CLI run view prints "cached" in place of
the duration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.tools_handler import ToolsHandler
from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.experimental.agent_executor import AgentExecutor
from crewai.hooks.tool_hooks import (
    clear_before_tool_call_hooks,
    register_before_tool_call_hook,
)
from crewai.llms.base_llm import BaseLLM
from crewai.tools.base_tool import BaseTool, to_langchain
from crewai.utilities.agent_utils import (
    convert_tools_to_openai_schema,
    execute_single_native_tool_call,
)

CALLS: list[str] = []
BLOCKED = "Tool execution blocked by hook. Tool: lookup"


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look a term up."

    def _run(self, term: str) -> str:
        CALLS.append(term)
        return f"value:{term}"


class _NativeToolCall:
    """An OpenAI-style native tool call, the format the executors receive."""

    class _Function:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    def __init__(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        self.id = call_id
        self.function = self._Function(name, json.dumps(args))


class _StubLLM(BaseLLM):
    """Enough of an LLM to build an Agent; no tool call here reaches a model."""

    def __init__(self) -> None:
        super().__init__(model="stub")

    def call(
        self,
        messages: str | list[Any],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
        **kwargs: Any,
    ) -> str:
        raise AssertionError("the probe never asks the model for anything")

    def supports_function_calling(self) -> bool:
        return True


def _executor(tool: BaseTool, cache: CacheHandler) -> CrewAgentExecutor:
    executor = CrewAgentExecutor(tools=to_langchain([tool]), original_tools=[tool])
    agent = Mock()
    agent.key = "test_agent"
    agent.role = "tester"
    agent.id = "agent-id"
    agent.verbose = False
    agent.fingerprint = None
    agent.tools_results = []
    task = Mock()
    task.name = "test"
    task.description = "test"
    task.id = "task-id"
    executor.agent = agent
    executor.task = task
    executor.tools_handler = ToolsHandler(cache=cache)
    return executor


def _call(executor: CrewAgentExecutor, tool: BaseTool, args: dict) -> tuple[dict, list]:
    _, available_functions, _ = convert_tools_to_openai_schema([tool])
    events: list[ToolUsageFinishedEvent] = []
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _capture(_source: object, event: ToolUsageFinishedEvent) -> None:
            events.append(event)

        result = executor._execute_single_native_tool_call(
            call_id="call_1",
            func_name="lookup",
            func_args=args,
            available_functions=available_functions,
            original_tool=tool,
        )
        crewai_event_bus.flush()
    return result, events


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_cached_native_tool_call_reports_from_cache_on_the_finished_event() -> None:
    CALLS.clear()
    tool = LookupTool()
    cache = CacheHandler()
    cache.add(tool="lookup", input=json.dumps({"term": "x"}), output="cached:x")

    result, events = _call(_executor(tool, cache), tool, {"term": "x"})

    assert result["from_cache"] is True
    assert CALLS == []
    assert [event.from_cache for event in events] == [True]
    assert events[0].output == "cached:x"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_live_native_tool_call_reports_from_cache_false() -> None:
    CALLS.clear()
    tool = LookupTool()

    result, events = _call(_executor(tool, CacheHandler()), tool, {"term": "y"})

    assert result["from_cache"] is False
    assert CALLS == ["y"]
    assert [event.from_cache for event in events] == [False]


def _default_executor(tool: BaseTool, cache: CacheHandler) -> AgentExecutor:
    """Build the executor `Agent.executor_class` actually defaults to."""
    from crewai.agent.core import Agent

    agent = Agent(
        role="tester",
        goal="Answer",
        backstory="You answer.",
        llm=_StubLLM(),
        verbose=False,
    )
    executor = AgentExecutor(
        agent=agent,
        llm=agent.llm,
        task=None,
        tools=to_langchain([tool]),
        original_tools=[tool],
    )
    executor.tools_handler = ToolsHandler(cache=cache)
    executor._setup_native_tools()
    return executor


def _with_a_blocking_before_tool_hook(run: Any) -> Any:
    """Run `run` with a `before_tool_call` hook that blocks every tool."""
    register_before_tool_call_hook(lambda _context: False)
    try:
        return run()
    finally:
        clear_before_tool_call_hooks()


def _capture_finished_events(
    run: Any,
) -> tuple[Any, list[ToolUsageFinishedEvent]]:
    events: list[ToolUsageFinishedEvent] = []
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _capture(_source: object, event: ToolUsageFinishedEvent) -> None:
            events.append(event)

        result = run()
        crewai_event_bus.flush()
    return result, events


def test_the_default_executor_reports_from_cache_on_a_replayed_native_call() -> None:
    CALLS.clear()
    tool = LookupTool()
    cache = CacheHandler()
    cache.add(tool="lookup", input=json.dumps({"term": "x"}), output="cached:x")
    executor = _default_executor(tool, cache)

    result, events = _capture_finished_events(
        lambda: executor._execute_single_native_tool_call(
            _NativeToolCall("call_1", "lookup", {"term": "x"})
        )
    )

    assert result["from_cache"] is True
    assert CALLS == []
    assert [event.from_cache for event in events] == [True]
    assert events[0].output == "cached:x"


def test_the_default_executor_reports_from_cache_false_on_a_live_native_call() -> None:
    CALLS.clear()
    tool = LookupTool()
    executor = _default_executor(tool, CacheHandler())

    result, events = _capture_finished_events(
        lambda: executor._execute_single_native_tool_call(
            _NativeToolCall("call_1", "lookup", {"term": "y"})
        )
    )

    assert result["from_cache"] is False
    assert CALLS == ["y"]
    assert [event.from_cache for event in events] == [False]


def test_the_default_executor_reports_no_cache_when_a_hook_blocks_a_cached_call() -> (
    None
):
    CALLS.clear()
    tool = LookupTool()
    cache = CacheHandler()
    cache.add(tool="lookup", input=json.dumps({"term": "x"}), output="cached:x")
    executor = _default_executor(tool, cache)

    result, events = _with_a_blocking_before_tool_hook(
        lambda: _capture_finished_events(
            lambda: executor._execute_single_native_tool_call(
                _NativeToolCall("call_1", "lookup", {"term": "x"})
            )
        )
    )

    assert result["result"] == BLOCKED
    assert result["from_cache"] is False
    assert CALLS == []
    assert [event.from_cache for event in events] == [False]
    assert events[0].output == BLOCKED


def _run_step_tool_call(tool: BaseTool, cache: CacheHandler, term: str) -> Any:
    """Run the helper `StepExecutor` uses for the default executor's steps."""
    _, available_functions, _ = convert_tools_to_openai_schema([tool])
    return execute_single_native_tool_call(
        _NativeToolCall("call_1", "lookup", {"term": term}),
        available_functions=available_functions,
        original_tools=[tool],
        structured_tools=None,
        tools_handler=ToolsHandler(cache=cache),
        agent=None,
        task=None,
        crew=None,
        event_source=object(),
    )


def test_the_step_executor_helper_reports_from_cache_on_a_replayed_native_call() -> (
    None
):
    CALLS.clear()
    tool = LookupTool()
    cache = CacheHandler()
    cache.add(tool="lookup", input=json.dumps({"term": "x"}), output="cached:x")

    result, events = _capture_finished_events(
        lambda: _run_step_tool_call(tool, cache, "x")
    )

    assert result.from_cache is True
    assert CALLS == []
    assert [event.from_cache for event in events] == [True]
    assert events[0].output == "cached:x"


def test_the_step_executor_helper_reports_from_cache_false_on_a_live_call() -> None:
    CALLS.clear()
    tool = LookupTool()

    result, events = _capture_finished_events(
        lambda: _run_step_tool_call(tool, CacheHandler(), "y")
    )

    assert result.from_cache is False
    assert CALLS == ["y"]
    assert [event.from_cache for event in events] == [False]


def test_the_step_executor_helper_reports_no_cache_when_a_hook_blocks_a_cached_call() -> (
    None
):
    CALLS.clear()
    tool = LookupTool()
    cache = CacheHandler()
    cache.add(tool="lookup", input=json.dumps({"term": "x"}), output="cached:x")

    result, events = _with_a_blocking_before_tool_hook(
        lambda: _capture_finished_events(lambda: _run_step_tool_call(tool, cache, "x"))
    )

    assert result.result == BLOCKED
    assert result.from_cache is False
    assert CALLS == []
    assert [event.from_cache for event in events] == [False]
    assert events[0].output == BLOCKED
