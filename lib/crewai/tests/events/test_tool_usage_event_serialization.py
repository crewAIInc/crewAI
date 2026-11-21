import pytest

from crewai.events.types.tool_usage_events import (
    ToolSelectionErrorEvent,
    ToolUsageErrorEvent,
    ToolValidateInputErrorEvent,
)


@pytest.mark.parametrize(
    "event_class",
    [ToolUsageErrorEvent, ToolValidateInputErrorEvent, ToolSelectionErrorEvent],
)
def test_error_event_with_exception_dumps_to_json(event_class):
    event = event_class(
        tool_name="check",
        tool_args={},
        error=ValueError("service unavailable"),
    )

    dumped = event.model_dump(mode="json")

    assert dumped["error"] == "service unavailable"


def test_error_event_keeps_non_exception_error_as_is():
    event = ToolUsageErrorEvent(tool_name="check", tool_args={}, error="boom")

    assert event.model_dump(mode="json")["error"] == "boom"
