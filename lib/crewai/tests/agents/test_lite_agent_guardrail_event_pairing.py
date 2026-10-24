"""Regression tests: LiteAgent guardrail retries must not re-emit the
execution started event.

``_execute_core`` used to recurse into itself for each guardrail retry, and
the started event is emitted at the top of the function, so every retry leaked
an extra unpaired ``LiteAgentExecutionStartedEvent`` (N retries -> N+1 started
events with a single terminal event). Trace listeners open one scope per
started event, so scopes leaked per retry.
"""

from collections import defaultdict
from typing import Any

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
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


@pytest.fixture
def execution_events() -> dict[str, list[Any]]:
    events: dict[str, list[Any]] = defaultdict(list)

    @crewai_event_bus.on(LiteAgentExecutionStartedEvent)
    def capture_started(_source: Any, event: Any) -> None:
        events["started"].append(event)

    @crewai_event_bus.on(LiteAgentExecutionCompletedEvent)
    def capture_completed(_source: Any, event: Any) -> None:
        events["completed"].append(event)

    @crewai_event_bus.on(LiteAgentExecutionErrorEvent)
    def capture_error(_source: Any, event: Any) -> None:
        events["error"].append(event)

    yield events

    crewai_event_bus.off(LiteAgentExecutionStartedEvent, capture_started)
    crewai_event_bus.off(LiteAgentExecutionCompletedEvent, capture_completed)
    crewai_event_bus.off(LiteAgentExecutionErrorEvent, capture_error)


def test_guardrail_retry_emits_single_started_and_completed(
    execution_events: dict[str, list[Any]],
) -> None:
    """Retries stay inside one execution scope: exactly one started and one
    terminal event per kickoff (fails on main: 3 started for 2 retries)."""
    calls = {"n": 0}

    def fail_twice_then_pass(output: Any) -> tuple[bool, str | Any]:
        calls["n"] += 1
        if calls["n"] <= 2:
            return (False, "not good enough")
        return (True, output)

    agent = LiteAgent(
        role="T",
        goal="G",
        backstory="B",
        llm=_OfflineLLM(),
        guardrail=fail_twice_then_pass,
        guardrail_max_retries=3,
    )

    result = agent.kickoff("hello")
    crewai_event_bus.flush()

    assert calls["n"] == 3, "guardrail should have retried twice then passed"
    assert result is not None
    assert len(execution_events["started"]) == 1
    assert len(execution_events["completed"]) == 1
    assert execution_events["error"] == []


def test_guardrail_exhausted_retries_pair_started_with_error(
    execution_events: dict[str, list[Any]],
) -> None:
    """Exhausted retries raise with one started + one error event (fails on
    main: retries leak unpaired started events on the failure path too)."""

    def always_fail(output: Any) -> tuple[bool, str]:
        return (False, "never good enough")

    agent = LiteAgent(
        role="T",
        goal="G",
        backstory="B",
        llm=_OfflineLLM(),
        guardrail=always_fail,
        guardrail_max_retries=2,
    )

    with pytest.raises(Exception, match="guardrail failed validation"):
        agent.kickoff("hello")
    crewai_event_bus.flush()

    assert len(execution_events["started"]) == 1
    assert len(execution_events["error"]) == 1
    assert execution_events["completed"] == []


def test_guardrail_pass_on_first_try_single_pair(
    execution_events: dict[str, list[Any]],
) -> None:
    """Guard: a guardrail that passes immediately keeps the single pair."""

    def always_pass(output: Any) -> tuple[bool, Any]:
        return (True, output)

    agent = LiteAgent(
        role="T",
        goal="G",
        backstory="B",
        llm=_OfflineLLM(),
        guardrail=always_pass,
        guardrail_max_retries=3,
    )

    result = agent.kickoff("hello")
    crewai_event_bus.flush()

    assert result is not None
    assert len(execution_events["started"]) == 1
    assert len(execution_events["completed"]) == 1
    assert execution_events["error"] == []
