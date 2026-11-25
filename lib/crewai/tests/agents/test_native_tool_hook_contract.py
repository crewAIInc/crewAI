"""Native tool hooks must retain aborts and use effective cache inputs."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from crewai import Agent
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.tools_handler import ToolsHandler
from crewai.events import crewai_event_bus
from crewai.agents.agent_executor import AgentExecutor
from crewai.hooks.dispatch import HookAborted, InterceptionPoint, clear_all, on
from crewai.hooks.tool_hooks import ToolCallHookContext
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from crewai.tools.base_tool import to_langchain
from crewai.utilities.agent_utils import (
    convert_tools_to_openai_schema,
    execute_single_native_tool_call,
)


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."
    calls: list[str] = []

    def _run(self, term: str = "default") -> str:
        self.calls.append(term)
        return f"value:{term}"


class ScriptedLLM(BaseLLM):
    def __init__(self, batch_size: int) -> None:
        super().__init__(model="offline")
        self.batch_size = batch_size
        self.calls = 0

    def supports_function_calling(self) -> bool:
        return True

    def call(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"term": str(i)}),
                    },
                }
                for i in range(self.batch_size)
            ]
        return "Finished"

    async def acall(self, messages: Any, **kwargs: Any) -> Any:
        return self.call(messages, **kwargs)


@pytest.fixture(autouse=True)
def clean_hooks():
    clear_all()
    yield
    crewai_event_bus.flush()
    clear_all()


@pytest.mark.parametrize("batch_size", [1, 2], ids=["serial", "parallel"])
@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_post_tool_abort_reaches_caller_without_another_model_call(
    batch_size, async_entry
):
    abort = HookAborted("policy denied", source="test-policy")

    @on(InterceptionPoint.POST_TOOL_CALL)
    def deny(_context):
        raise abort

    llm = ScriptedLLM(batch_size)
    agent = Agent(
        role="Tester",
        goal="Look up values",
        backstory="Test hooks",
        llm=llm,
        tools=[LookupTool()],
        max_retry_limit=0,
        verbose=False,
    )

    with pytest.raises(HookAborted) as exc:
        if async_entry:
            asyncio.run(agent.kickoff_async("Look up values"))
        else:
            agent.kickoff("Look up values")

    assert exc.value is abort
    assert llm.calls == 1


@pytest.mark.parametrize("batch_size", [1, 2], ids=["serial", "parallel"])
@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_kickoff_history_records_input_applied_by_pre_tool_hooks(
    batch_size: int, async_entry: bool
) -> None:
    class LookupWithExecutionNote(LookupTool):
        execution_notes: list[str] = []

        def _run(self, term: str = "default", execution_note: str = "") -> str:
            self.execution_notes.append(execution_note)
            return super()._run(term)

    @on(InterceptionPoint.PRE_TOOL_CALL)
    def rewrite(context: ToolCallHookContext) -> None:
        context.tool_input["term"] = f"effective-{context.tool_input['term']}"
        context.tool_input["execution_note"] = "for the tool only"

    @on(InterceptionPoint.POST_TOOL_CALL)
    def edit_context_after_execution(context: ToolCallHookContext) -> None:
        context.tool_input["term"] = "after-execution"

    tool = LookupWithExecutionNote()
    agent = Agent(
        role="Tester",
        goal="Look up values",
        backstory="Test hooks",
        llm=ScriptedLLM(batch_size),
        tools=[tool],
        verbose=False,
    )

    output = (
        asyncio.run(agent.kickoff_async("Look up values"))
        if async_entry
        else agent.kickoff("Look up values")
    )

    expected = [f"effective-{i}" for i in range(batch_size)]
    assert sorted(tool.calls) == expected
    assert tool.execution_notes == ["for the tool only"] * batch_size
    calls = [call for message in output.messages for call in message.get("tool_calls", [])]
    assert [call["id"] for call in calls] == [f"call_{i}" for i in range(batch_size)]
    assert [json.loads(call["function"]["arguments"]) for call in calls] == [
        {"term": term} for term in expected
    ]


def native_call(backend, tool, cache, args):
    structured = to_langchain([tool])
    _, functions, _ = convert_tools_to_openai_schema([tool])
    handler = ToolsHandler(cache=cache)
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps(args)},
    }
    if backend == "shared":
        result = execute_single_native_tool_call(
            call,
            available_functions=functions,
            original_tools=[tool],
            structured_tools=structured,
            tools_handler=handler,
            agent=None,
            task=None,
            crew=None,
            event_source=tool,
        )
        return result.result, result.from_cache
    executor = AgentExecutor(
        tools=structured, original_tools=[tool], tools_handler=handler
    )
    executor._setup_native_tools()
    result = executor._execute_single_native_tool_call(call)
    return result["result"], result["from_cache"]


def test_non_object_tool_arguments_still_fail_event_validation() -> None:
    with pytest.raises(ValidationError, match="ToolUsageStartedEvent"):
        native_call("agent", LookupTool(), CacheHandler(), 5)


@pytest.mark.parametrize("backend", ["agent", "shared"])
@pytest.mark.parametrize(
    "input_args", [{"term": "original"}, {}], ids=["original", "empty"]
)
@pytest.mark.parametrize("seed", ["miss", "original", "effective"])
def test_cache_identity_uses_arguments_after_hooks(backend, input_args, seed):
    @on(InterceptionPoint.PRE_TOOL_CALL)
    def rewrite(context):
        context.tool_input["term"] = "effective"

    tool = LookupTool()
    cache = CacheHandler()
    original_key = json.dumps(input_args) if input_args else ""
    effective_key = json.dumps({"term": "effective"})
    if seed != "miss":
        cache.add("lookup", original_key, "stale:original")
    if seed == "effective":
        cache.add("lookup", effective_key, "cached:effective")

    result, from_cache = native_call(backend, tool, cache, dict(input_args))

    assert result == ("cached:effective" if seed == "effective" else "value:effective")
    assert from_cache is (seed == "effective")
    assert tool.calls == ([] if seed == "effective" else ["effective"])
    assert cache.read("lookup", effective_key) == result
    assert cache.read("lookup", original_key) == (
        None if seed == "miss" else "stale:original"
    )


@pytest.mark.parametrize("backend", ["agent", "shared"])
def test_blocked_hook_does_not_return_a_cached_result(backend):
    @on(InterceptionPoint.PRE_TOOL_CALL)
    def block(_context):
        return False

    tool = LookupTool()
    cache = CacheHandler()
    key = json.dumps({"term": "original"})
    cache.add("lookup", key, "cached:original")

    result, from_cache = native_call(backend, tool, cache, {"term": "original"})

    assert result == "Tool execution blocked by hook. Tool: lookup"
    assert from_cache is False
    assert tool.calls == []
    assert cache.read("lookup", key) == "cached:original"


@pytest.mark.parametrize("backend", ["agent", "shared"])
def test_finished_event_reports_a_cache_hit(backend):
    """The finished event should say the cache answered, not that the tool ran."""
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

    tool = LookupTool()
    cache = CacheHandler()
    cache.add("lookup", json.dumps({"term": "original"}), "cached:original")

    finished: list[ToolUsageFinishedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _collect(_source, event):
            finished.append(event)

        result, from_cache = native_call(backend, tool, cache, {"term": "original"})
        crewai_event_bus.flush()

    assert from_cache is True
    assert tool.calls == []
    assert [event.from_cache for event in finished] == [True]
    assert finished[0].output == result
