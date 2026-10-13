"""A native function-calling tool call served from the cache says so on its event.

`_execute_single_native_tool_call` reads the tools handler's cache and skips
the tool body on a hit, but its `ToolUsageFinishedEvent` used to be emitted
without `from_cache`, so every consumer of the bus (tracing included) saw a
replayed call as a live one. The text-protocol path (`ToolUsage`) already
carried the flag.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.tools_handler import ToolsHandler
from crewai.events import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.tools.base_tool import BaseTool, to_langchain
from crewai.utilities.agent_utils import convert_tools_to_openai_schema

CALLS: list[str] = []


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look a term up."

    def _run(self, term: str) -> str:
        CALLS.append(term)
        return f"value:{term}"


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
