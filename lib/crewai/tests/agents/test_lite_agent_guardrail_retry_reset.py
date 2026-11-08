"""Regression tests: the LiteAgent guardrail retry counter must reset per
kickoff.

``kickoff`` reset ``_iterations``/``tools_results``/``_tool_failures`` but not
``_guardrail_retry_count``, so a reused agent instance inherited the previous
run's consumed budget: after exhausting retries once, the next kickoff raised
on its first guardrail failure instead of getting its own fresh budget.
"""

from typing import Any

import pytest

from crewai.lite_agent import LiteAgent
from crewai.llms.base_llm import BaseLLM


class _OfflineLLM(BaseLLM):
    """Offline LLM that finishes immediately without any network call."""

    def __init__(self) -> None:
        super().__init__(model="offline-fake-model")

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        return "Thought: done.\nFinal Answer: the answer"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096


def _make_agent(guardrail: Any) -> LiteAgent:
    return LiteAgent(
        role="T",
        goal="G",
        backstory="B",
        llm=_OfflineLLM(),
        guardrail=guardrail,
        guardrail_max_retries=3,
    )


def test_retry_counter_resets_between_kickoffs() -> None:
    """A reused agent gets a fresh retry budget per kickoff (fails on main:
    the second run raises on its first guardrail failure)."""
    calls = {"n": 0}

    def fail_twice_then_pass(output: Any) -> tuple[bool, str | Any]:
        calls["n"] += 1
        if calls["n"] % 3 in (1, 2):  # fail the first two calls of each run
            return (False, "not good enough")
        return (True, output)

    agent = _make_agent(fail_twice_then_pass)

    first = agent.kickoff("run one")
    assert first is not None
    assert calls["n"] == 3  # 2 failed attempts + 1 passing

    calls["n"] = 0
    second = agent.kickoff("run two")  # on main: raises after 1 attempt
    assert second is not None
    assert calls["n"] == 3, "second kickoff should get its own full budget"


def test_exhausted_budget_does_not_leak_into_next_kickoff() -> None:
    """After one kickoff exhausts its retries, the next kickoff still gets the
    full budget (fails on main: the next kickoff gets a single attempt)."""
    calls = {"n": 0}

    def always_fail(output: Any) -> tuple[bool, str]:
        calls["n"] += 1
        return (False, "never good enough")

    agent = _make_agent(always_fail)

    with pytest.raises(Exception, match="guardrail failed validation"):
        agent.kickoff("run A")
    assert calls["n"] == 4  # initial attempt + 3 retries

    calls["n"] = 0
    with pytest.raises(Exception, match="guardrail failed validation"):
        agent.kickoff("run B")
    assert calls["n"] == 4, "each kickoff should exhaust its own budget"
