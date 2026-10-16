from collections.abc import Callable
from functools import partial
import json
from types import FunctionType

from crewai.events.types.llm_guardrail_events import LLMGuardrailStartedEvent
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail import process_guardrail
import pytest


def validate_output(output: TaskOutput) -> tuple[bool, TaskOutput]:
    return True, output


class CallableGuardrail:
    def __call__(self, output: TaskOutput) -> tuple[bool, TaskOutput]:
        return True, output


@pytest.mark.parametrize(
    ("guardrail", "expected_guardrail"),
    [
        (
            FunctionType(
                validate_output.__code__.replace(co_filename="<dynamic-guardrail>"),
                globals(),
            ),
            "validate_output",
        ),
        (partial(validate_output), "partial"),
        (CallableGuardrail(), "CallableGuardrail"),
    ],
    ids=["dynamic-function", "partial", "callable-instance"],
)
def test_guardrail_runs_without_inspectable_source(
    guardrail: Callable[[TaskOutput], tuple[bool, TaskOutput]],
    expected_guardrail: str,
) -> None:
    output = TaskOutput(description="Validate output", raw="done", agent="test")

    result = process_guardrail(output, guardrail, retry_count=0)

    assert result.success is True
    assert result.result is output
    event = LLMGuardrailStartedEvent(guardrail=guardrail, retry_count=0)
    assert event.guardrail_type == "function"
    assert event.guardrail == expected_guardrail
    assert json.loads(event.model_dump_json())["guardrail"] == event.guardrail


def test_guardrail_event_keeps_available_source() -> None:
    event = LLMGuardrailStartedEvent(guardrail=validate_output, retry_count=0)

    assert event.guardrail_name == "validate_output"
    assert "return True, output" in event.guardrail
