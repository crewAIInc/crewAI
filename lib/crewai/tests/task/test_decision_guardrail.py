"""Tests for DecisionGuardrail, the threshold-based predicate guardrail."""

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crewai.agent import Agent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.hooks.dispatch import HookAborted
from crewai.task import Task
from crewai.tasks.decision_guardrail import DecisionGuardrail, PredicateAnswer
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail import process_guardrail


INSTRUCTIONS = "The output greets the reader"


class FakeDecider:
    """Answers predicate questions from a script, recording each call.

    Script items are probabilities (answered under the asked name), any other
    answer object (returned as-is), or exceptions (raised).
    """

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict[str, str]] = []

    def predicate(
        self, input: str, *, name: str, instructions: str
    ) -> PredicateAnswer | None:
        self.calls.append({"input": input, "name": name, "instructions": instructions})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, float):
            return PredicateAnswer(name=name, probability=item)
        return item


def make_guardrail(*script: Any, threshold: float = 0.5) -> DecisionGuardrail:
    return DecisionGuardrail(INSTRUCTIONS, FakeDecider(*script), threshold=threshold)


@pytest.fixture
def task_output() -> TaskOutput:
    return TaskOutput(
        description="Greet",
        raw="Hello there",
        agent="Greeter",
        expected_output="A greeting",
    )


def mock_agent(*results: str) -> Mock:
    agent = Mock()
    agent.role = "test_agent"
    agent.crew = None
    agent.last_messages = []
    agent.execute_task.side_effect = list(results)
    return agent


# --- DecisionGuardrail called directly ---------------------------------------


def test_passes_and_keeps_original_output_above_threshold(task_output):
    assert make_guardrail(0.9)(task_output) == (True, "Hello there")


def test_fails_below_threshold_with_probability_in_message(task_output):
    success, message = make_guardrail(0.3)(task_output)

    assert success is False
    assert "0.300" in message
    assert "0.500" in message


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_equality_boundary(task_output, threshold):
    assert make_guardrail(threshold, threshold=threshold)(task_output)[0] is True
    if threshold > 0.0:
        just_below = math.nextafter(threshold, 0.0)
        assert make_guardrail(just_below, threshold=threshold)(task_output)[0] is False


@pytest.mark.parametrize(
    "threshold", [-0.1, 1.1, math.nan, math.inf, True, "0.5", None]
)
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        DecisionGuardrail(INSTRUCTIONS, FakeDecider(0.9), threshold=threshold)


@pytest.mark.parametrize("instructions", ["", "   "])
def test_rejects_empty_instructions(instructions):
    with pytest.raises(ValueError, match="instructions"):
        DecisionGuardrail(instructions, FakeDecider(0.9), threshold=0.5)


def test_missing_answer_fails_closed(task_output):
    success, message = make_guardrail(None)(task_output)

    assert success is False
    assert "No decision" in message


def test_mismatched_answer_name_fails_closed(task_output):
    other = PredicateAnswer(name="some_other_question", probability=1.0)

    success, message = make_guardrail(other)(task_output)

    assert success is False
    assert "some_other_question" in message


@pytest.mark.parametrize(
    "answer",
    [
        SimpleNamespace(name="task_output_check", probability=math.nan),
        SimpleNamespace(name="task_output_check", probability=-0.01),
        SimpleNamespace(name="task_output_check", probability=1.01),
        SimpleNamespace(name="task_output_check", probability="0.9"),
        SimpleNamespace(name="task_output_check", probability=None),
        SimpleNamespace(name="task_output_check", probability=True),
        SimpleNamespace(name="task_output_check"),
    ],
)
def test_malformed_probability_fails_closed(task_output, answer):
    guardrail = make_guardrail(answer, threshold=0.0)

    success, _ = guardrail(task_output)

    assert success is False
    assert guardrail.last_answer is None


def test_decider_error_fails_closed(task_output):
    success, message = make_guardrail(RuntimeError("service unavailable"))(task_output)

    assert success is False
    assert "service unavailable" in message


def test_hook_aborted_propagates(task_output):
    guardrail = make_guardrail(HookAborted(reason="denied by policy"))

    with pytest.raises(HookAborted, match="denied by policy"):
        guardrail(task_output)


def test_last_answer_tracks_latest_evaluation(task_output):
    guardrail = make_guardrail(0.9, 0.2, None, RuntimeError("boom"))

    guardrail(task_output)
    assert guardrail.last_answer == PredicateAnswer(
        name="task_output_check", probability=0.9
    )
    guardrail(task_output)
    assert guardrail.last_answer is not None
    assert guardrail.last_answer.probability == 0.2
    guardrail(task_output)
    assert guardrail.last_answer is None
    guardrail(task_output)
    assert guardrail.last_answer is None


def test_decider_receives_raw_output_name_and_instructions(task_output):
    decider = FakeDecider(0.9)
    guardrail = DecisionGuardrail(
        INSTRUCTIONS, decider, threshold=0.5, name="greets_reader"
    )

    guardrail(task_output)

    assert decider.calls == [
        {"input": "Hello there", "name": "greets_reader", "instructions": INSTRUCTIONS}
    ]


# --- Through Task and process_guardrail --------------------------------------


def test_task_accepts_decision_guardrail_and_keeps_output():
    decider = FakeDecider(0.9)
    task = Task(
        description="Greet",
        expected_output="A greeting",
        guardrail=DecisionGuardrail(INSTRUCTIONS, decider, threshold=0.5),
    )

    result = task.execute_sync(agent=mock_agent("Hello there"))

    assert result.raw == "Hello there"
    assert len(decider.calls) == 1


def test_task_retries_after_low_probability_then_passes():
    decider = FakeDecider(0.2, 0.9)
    agent = mock_agent("bad", "good")
    task = Task(
        description="Greet",
        expected_output="A greeting",
        guardrail=DecisionGuardrail(INSTRUCTIONS, decider, threshold=0.5),
    )

    result = task.execute_sync(agent=agent)

    assert result.raw == "good"
    assert task.retry_count == 1
    assert [call["input"] for call in decider.calls] == ["bad", "good"]
    retry_context = agent.execute_task.call_args_list[1].kwargs["context"]
    assert "probability 0.200 < required 0.500" in retry_context


def test_task_raises_when_retries_exhausted():
    decider = FakeDecider(0.1)
    task = Task(
        description="Greet",
        expected_output="A greeting",
        guardrail=DecisionGuardrail(INSTRUCTIONS, decider, threshold=0.5),
        guardrail_max_retries=1,
    )

    with pytest.raises(Exception, match="validation after 1 retries") as exc_info:
        task.execute_sync(agent=mock_agent("bad", "still bad"))

    assert "probability 0.100" in str(exc_info.value)
    assert len(decider.calls) == 2


def test_task_never_approves_when_decider_keeps_failing():
    decider = FakeDecider(RuntimeError("service unavailable"))
    task = Task(
        description="Greet",
        expected_output="A greeting",
        guardrail=DecisionGuardrail(INSTRUCTIONS, decider, threshold=0.5),
        guardrail_max_retries=2,
    )

    with pytest.raises(Exception, match="service unavailable"):
        task.execute_sync(agent=mock_agent("one", "two", "three"))

    assert len(decider.calls) == 3


def test_started_event_classifies_decision_guardrail(task_output):
    started: list[LLMGuardrailStartedEvent] = []
    completed: list[LLMGuardrailCompletedEvent] = []
    guardrail = DecisionGuardrail(
        INSTRUCTIONS, FakeDecider(0.9), threshold=0.5, name="greets_reader"
    )

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def _on_started(_source: Any, event: LLMGuardrailStartedEvent) -> None:
            started.append(event)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def _on_completed(_source: Any, event: LLMGuardrailCompletedEvent) -> None:
            completed.append(event)

        process_guardrail(output=task_output, guardrail=guardrail, retry_count=0)
        crewai_event_bus.flush(timeout=5)

    assert len(started) == 1
    assert started[0].guardrail_type == "decision"
    assert started[0].guardrail_name == "greets_reader"
    assert started[0].guardrail == f"Decision predicate: {INSTRUCTIONS}"
    assert len(completed) == 1
    assert completed[0].success is True


@pytest.mark.asyncio
@patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
async def test_async_task_with_decision_guardrail(mock_execute: AsyncMock) -> None:
    mock_execute.return_value = "Hello there"
    agent = Agent(role="Greeter", goal="Greet", backstory="Friendly", llm="gpt-4o-mini")
    task = Task(
        description="Greet",
        expected_output="A greeting",
        agent=agent,
        guardrail=DecisionGuardrail(INSTRUCTIONS, FakeDecider(0.9), threshold=0.5),
    )

    result = await task.aexecute_sync()

    assert result.raw == "Hello there"
