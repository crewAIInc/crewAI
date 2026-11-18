"""A guardrail that raises owes its started event a terminal one.

``process_guardrail`` emits ``LLMGuardrailStartedEvent`` before calling the
guardrail and ``LLMGuardrailCompletedEvent`` after it returns. A deny
(``HookAborted``) already gets its terminal event; an ordinary exception did
not, so the tracing listener's "evaluate guardrail" span stayed open until the
end-of-execution sweep closed it as "orphaned", losing the real error from the
trace. The caller sees the exception either way — only the event pairing
differs.
"""

from __future__ import annotations

from typing import Any

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_guardrail_events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail import process_guardrail

from tests.utils import wait_for_event_handlers


def test_a_raising_guardrail_still_reports_the_validation_it_started():
    task_output = TaskOutput(
        description="Say hi", raw="hi", agent="Doer", expected_output="A greeting."
    )

    def raising_guardrail(_output: Any) -> tuple[bool, str]:
        raise ValueError("guardrail exploded")

    started: list[LLMGuardrailStartedEvent] = []
    completed: list[LLMGuardrailCompletedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def _on_started(_source: Any, event: LLMGuardrailStartedEvent) -> None:
            started.append(event)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def _on_completed(_source: Any, event: LLMGuardrailCompletedEvent) -> None:
            completed.append(event)

        with pytest.raises(ValueError, match="guardrail exploded"):
            process_guardrail(
                output=task_output, guardrail=raising_guardrail, retry_count=0
            )

        wait_for_event_handlers()

    assert len(started) == 1
    assert len(completed) == 1
    assert completed[0].success is False
    assert "guardrail exploded" in (completed[0].error or "")


def test_a_raising_guardrail_keeps_the_exception_for_the_caller():
    task_output = TaskOutput(
        description="Say hi", raw="hi", agent="Doer", expected_output="A greeting."
    )

    def raising_guardrail(_output: Any) -> tuple[bool, str]:
        raise RuntimeError("context lost")

    with crewai_event_bus.scoped_handlers():
        with pytest.raises(RuntimeError, match="context lost"):
            process_guardrail(
                output=task_output, guardrail=raising_guardrail, retry_count=1
            )
        wait_for_event_handlers()


@pytest.mark.parametrize(
    ("returned", "error_type"),
    [((False, 42), "ValidationError"), ((True, "hi", "extra"), "ValueError")],
    ids=["non-string-error", "three-tuple"],
)
def test_a_malformed_guardrail_result_still_reports_the_validation_it_started(
    returned: Any, error_type: str
):
    task_output = TaskOutput(
        description="Say hi", raw="hi", agent="Doer", expected_output="A greeting."
    )

    def malformed_guardrail(_output: Any) -> Any:
        return returned

    started: list[LLMGuardrailStartedEvent] = []
    completed: list[LLMGuardrailCompletedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def _on_started(_source: Any, event: LLMGuardrailStartedEvent) -> None:
            started.append(event)

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def _on_completed(_source: Any, event: LLMGuardrailCompletedEvent) -> None:
            completed.append(event)

        with pytest.raises(ValueError):
            process_guardrail(
                output=task_output, guardrail=malformed_guardrail, retry_count=0
            )

        wait_for_event_handlers()

    assert len(started) == 1
    assert len(completed) == 1
    assert completed[0].success is False
    assert (completed[0].error or "").startswith(error_type)


def test_a_valid_guardrail_result_reports_one_completed_event():
    task_output = TaskOutput(
        description="Say hi", raw="hi", agent="Doer", expected_output="A greeting."
    )

    completed: list[LLMGuardrailCompletedEvent] = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def _on_completed(_source: Any, event: LLMGuardrailCompletedEvent) -> None:
            completed.append(event)

        result = process_guardrail(
            output=task_output, guardrail=lambda o: (False, "too short"), retry_count=0
        )

        wait_for_event_handlers()

    assert result.error == "too short"
    assert len(completed) == 1
    assert completed[0].success is False
    assert completed[0].error == "too short"
