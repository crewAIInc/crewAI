"""Native tool hooks must retain aborts and use effective cache inputs."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from crewai import Agent
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.tools_handler import ToolsHandler
from crewai.events import crewai_event_bus
from crewai.agents.agent_executor import AgentExecutor
from crewai.hooks.dispatch import HookAborted, InterceptionPoint, clear_all, on
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


class AnswerTool(BaseTool):
    name: str = "final_lookup"
    description: str = "Returns the final answer."
    result_as_answer: bool = True
    calls: list[str] = []

    def _run(self, term: str = "x") -> str:
        self.calls.append(term)
        return f"answer:{term}"


class ToolThenAnswerLLM(BaseLLM):
    """Asks for the tool once, then answers without any tool call."""

    def __init__(self) -> None:
        super().__init__(model="offline")
        self.calls = 0

    def supports_function_calling(self) -> bool:
        return True

    def call(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "final_lookup",
                        "arguments": json.dumps({"term": "q"}),
                    },
                }
            ]
        return "final answer after the block"

    async def acall(self, messages: Any, **kwargs: Any) -> Any:
        return self.call(messages, **kwargs)


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_blocked_result_as_answer_tool_does_not_become_final_answer(async_entry):
    @on(InterceptionPoint.PRE_TOOL_CALL)
    def block(_context):
        return False

    llm = ToolThenAnswerLLM()
    tool = AnswerTool()
    agent = Agent(
        role="Tester",
        goal="Look up values",
        backstory="Test hooks",
        llm=llm,
        tools=[tool],
        max_retry_limit=0,
        verbose=False,
    )

    if async_entry:
        output = asyncio.run(agent.kickoff_async("Look up values"))
    else:
        output = agent.kickoff("Look up values")

    # The block notice must reach the model as tool context, not end the run.
    assert "blocked by hook" not in str(output)
    assert "final answer after the block" in str(output)
    assert llm.calls == 2
    assert tool.calls == []


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_result_as_answer_tool_still_short_circuits_when_not_blocked(async_entry):
    llm = ToolThenAnswerLLM()
    tool = AnswerTool()
    agent = Agent(
        role="Tester",
        goal="Look up values",
        backstory="Test hooks",
        llm=llm,
        tools=[tool],
        max_retry_limit=0,
        verbose=False,
    )

    if async_entry:
        output = asyncio.run(agent.kickoff_async("Look up values"))
    else:
        output = agent.kickoff("Look up values")

    assert str(output) == "answer:q"
    assert llm.calls == 1
    assert tool.calls == ["q"]


def test_blocked_native_call_reports_hook_blocked_flag():
    @on(InterceptionPoint.PRE_TOOL_CALL)
    def block(_context):
        return False

    tool = AnswerTool()
    executor = AgentExecutor(
        tools=to_langchain([tool]),
        original_tools=[tool],
        tools_handler=ToolsHandler(cache=CacheHandler()),
    )
    executor._setup_native_tools()
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "final_lookup", "arguments": json.dumps({"term": "q"})},
    }

    result = executor._execute_single_native_tool_call(call)

    assert result["result"] == "Tool execution blocked by hook. Tool: final_lookup"
    assert result["hook_blocked"] is True
    assert result["tool_failure"] is None
