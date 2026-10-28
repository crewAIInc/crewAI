"""max_iter counts every tool call, including each of several parallel ones."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from crewai import Agent, Crew, Task
from crewai.events import crewai_event_bus
from crewai.hooks.dispatch import clear_all
from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    clear_all_llm_call_hooks,
    register_before_llm_call_hook,
)
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from crewai.utilities.i18n import I18N_DEFAULT


FORCE_FINAL_ANSWER = I18N_DEFAULT.errors("force_final_answer")
NOT_RUN = "Not run: the iteration limit (max_iter) was reached before this tool call."


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."
    calls: list[str] = []

    def _run(self, term: str = "default") -> str:
        self.calls.append(term)
        return f"value:{term}"


class SubmitTool(BaseTool):
    name: str = "submit"
    description: str = "Submit the final answer."
    result_as_answer: bool = True

    def _run(self, term: str = "default") -> str:
        return f"submitted:{term}"


class ScriptedLLM(BaseLLM):
    """Answers each turn from a script; the forced final answer gets its own reply.

    A script entry that is an int is a batch of that many ``lookup`` tool calls;
    a list is a batch calling those tools by name; a string is a final answer.
    """

    def __init__(
        self, script: list[int | list[str] | str], native: bool = True
    ) -> None:
        super().__init__(model="offline")
        self.script = list(script)
        self.native = native
        self.turns: list[list[dict[str, Any]]] = []
        self.batch_number = 0

    def supports_function_calling(self) -> bool:
        return self.native

    def call(self, messages: Any, **kwargs: Any) -> Any:
        self.turns.append(list(messages))
        if messages[-1]["content"] == FORCE_FINAL_ANSWER:
            return "Forced answer"
        step = self.script.pop(0)
        if isinstance(step, str):
            return step
        self.batch_number += 1
        names = ["lookup"] * step if isinstance(step, int) else step
        terms = [f"{self.batch_number}.{i}" for i in range(len(names))]
        if not self.native:
            return (
                "Thought: I need to look this up\n"
                "Action: lookup\n"
                f'Action Input: {{"term": "{terms[0]}"}}'
            )
        return [
            {
                "id": f"call_{term}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps({"term": term}),
                },
            }
            for name, term in zip(names, terms, strict=True)
        ]

    async def acall(self, messages: Any, **kwargs: Any) -> Any:
        return self.call(messages, **kwargs)


@pytest.fixture(autouse=True)
def clean_hooks():
    clear_all()
    clear_all_llm_call_hooks()
    yield
    crewai_event_bus.flush()
    clear_all()
    clear_all_llm_call_hooks()


def _run(
    llm: ScriptedLLM,
    tool: BaseTool,
    max_iter: int,
    async_entry: bool,
    extra_tools: list[BaseTool] | None = None,
) -> str:
    agent = Agent(
        role="Investigator",
        goal="Look up values",
        backstory="Looks things up.",
        llm=llm,
        tools=[tool, *(extra_tools or [])],
        max_iter=max_iter,
        max_retry_limit=0,
        verbose=False,
    )
    if async_entry:
        output = asyncio.run(agent.kickoff_async("Look up values"))
    else:
        output = agent.kickoff("Look up values")
    return output.raw


def _tool_messages(messages: list[dict[str, Any]]) -> dict[str, str]:
    return {
        m["tool_call_id"]: m["content"] for m in messages if m.get("role") == "tool"
    }


@pytest.fixture
def lookup() -> LookupTool:
    return LookupTool(calls=[])


@pytest.fixture
def sequential_lookup() -> LookupTool:
    # A usage cap makes the executor run a batch sequentially, not in parallel.
    return LookupTool(calls=[], max_usage_count=100)


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("tool_kind", ["parallel", "sequential"])
def test_batch_larger_than_max_iter_runs_only_the_budget_then_forces_answer(
    async_entry, tool_kind, lookup, sequential_lookup
):
    tool = lookup if tool_kind == "parallel" else sequential_lookup
    llm = ScriptedLLM([5, "Never reached"])

    result = _run(llm, tool, max_iter=3, async_entry=async_entry)

    assert result == "Forced answer"
    assert sorted(tool.calls) == ["1.0", "1.1", "1.2"]
    forced_turn = llm.turns[-1]
    assert forced_turn[-1]["content"] == FORCE_FINAL_ANSWER
    results = _tool_messages(forced_turn)
    assert results["call_1.0"] == "value:1.0"
    assert results["call_1.2"] == "value:1.2"
    assert results["call_1.3"].startswith(NOT_RUN)
    assert results["call_1.4"].startswith(NOT_RUN)
    assert len(llm.turns) == 2


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_tool_calls_accumulate_across_steps(async_entry, lookup):
    llm = ScriptedLLM([3, 3, "Never reached"])

    result = _run(llm, lookup, max_iter=5, async_entry=async_entry)

    assert result == "Forced answer"
    assert sorted(lookup.calls) == ["1.0", "1.1", "1.2", "2.0", "2.1"]
    results = _tool_messages(llm.turns[-1])
    assert results["call_2.2"].startswith(NOT_RUN)
    assert len(llm.turns) == 3


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_batches_within_max_iter_run_in_full_and_the_model_answers(
    async_entry, lookup
):
    llm = ScriptedLLM([3, 2, "Finished"])

    result = _run(llm, lookup, max_iter=25, async_entry=async_entry)

    assert result == "Finished"
    assert sorted(lookup.calls) == ["1.0", "1.1", "1.2", "2.0", "2.1"]
    final_turn = llm.turns[-1]
    assert FORCE_FINAL_ANSWER not in [m["content"] for m in final_turn]
    assert not any(c.startswith(NOT_RUN) for c in _tool_messages(final_turn).values())


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_exactly_max_iter_tool_calls_then_answer_is_forced(async_entry, lookup):
    llm = ScriptedLLM([3, "Never reached"])

    result = _run(llm, lookup, max_iter=3, async_entry=async_entry)

    assert result == "Forced answer"
    assert sorted(lookup.calls) == ["1.0", "1.1", "1.2"]
    assert not any(
        c.startswith(NOT_RUN) for c in _tool_messages(llm.turns[-1]).values()
    )


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_text_tool_path_still_counts_one_per_tool_call(async_entry, lookup):
    llm = ScriptedLLM([1, 1, 1, "Final Answer: never reached"], native=False)

    result = _run(llm, lookup, max_iter=2, async_entry=async_entry)

    assert result == "Forced answer"
    assert lookup.calls == ["1.0", "2.0"]


def test_llm_hook_iterations_count_tool_calls(lookup):
    seen: list[int] = []

    def record(context: LLMCallHookContext) -> None:
        seen.append(context.iterations)

    register_before_llm_call_hook(record)
    llm = ScriptedLLM([3, 1, "Finished"])

    result = _run(llm, lookup, max_iter=25, async_entry=False)

    assert result == "Finished"
    assert seen == [0, 3, 4]


def test_crew_task_counts_parallel_tool_calls(lookup):
    llm = ScriptedLLM([5, "Never reached"])
    agent = Agent(
        role="Investigator",
        goal="Look up values",
        backstory="Looks things up.",
        llm=llm,
        tools=[lookup],
        max_iter=3,
        max_retry_limit=0,
        verbose=False,
    )
    task = Task(description="Look up values", expected_output="Values", agent=agent)

    result = Crew(agents=[agent], tasks=[task]).kickoff()

    assert result.raw == "Forced answer"
    assert sorted(lookup.calls) == ["1.0", "1.1", "1.2"]


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_result_as_answer_call_within_the_budget_still_ends_the_run(
    async_entry, lookup
):
    llm = ScriptedLLM([["lookup", "submit", "lookup"], "Never reached"])

    result = _run(llm, lookup, 3, async_entry, extra_tools=[SubmitTool()])

    assert result == "submitted:1.1"
    assert lookup.calls == ["1.0"]
    assert len(llm.turns) == 1


@pytest.mark.parametrize("async_entry", [False, True], ids=["sync", "async"])
def test_result_as_answer_call_past_the_budget_is_not_run(async_entry, lookup):
    # The limit is hard: a submit call past it is not run, the same as a submit
    # made in a later turn once max_iter is used up. The answer is then forced.
    llm = ScriptedLLM([["lookup", "lookup", "submit"], "Never reached"])

    result = _run(llm, lookup, 2, async_entry, extra_tools=[SubmitTool()])

    assert result == "Forced answer"
    assert sorted(lookup.calls) == ["1.0", "1.1"]
    assert _tool_messages(llm.turns[-1])["call_1.2"].startswith(NOT_RUN)
    assert len(llm.turns) == 2
