"""A `@human_feedback` gate must reach the trace as more than method start/finish.

The listener subscribed to method and conversation events but not to the
review-gate events (`HumanFeedbackRequestedEvent`, `HumanFeedbackReceivedEvent`)
or the pause events (`MethodExecutionPausedEvent`, `FlowPausedEvent`). A trace
could therefore not say that a run stopped for review, what the reviewer was
shown, or what they answered. Each is collected as a whole-event payload, like
every other non-complex type.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from unittest.mock import patch

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.listeners.tracing.types import TraceEvent
from crewai.events.types.flow_events import (
    FlowPausedEvent,
    HumanFeedbackReceivedEvent,
    HumanFeedbackRequestedEvent,
    MethodExecutionPausedEvent,
)
import pytest


@pytest.fixture
def listener() -> Iterator[TraceCollectionListener]:
    """A listener with tracing enabled and every handler registered.

    Enabled through the environment like the other tracing tests. Only the
    env var is set, not the tracing context var, so batch initialization
    never reaches the backend: `_initialize_backend_batch` returns before
    any request when tracing is not enabled in context.

    `scoped_handlers` is required: the bus is a process-wide singleton, so
    handlers registered here would otherwise fire in whatever runs next.
    """
    with (
        crewai_event_bus.scoped_handlers(),
        patch.dict(os.environ, {"CREWAI_TRACING_ENABLED": "true"}),
    ):
        trace_listener = TraceCollectionListener()
        trace_listener.setup_listeners(crewai_event_bus)
        yield trace_listener


def _collected(listener: TraceCollectionListener, event_type: str) -> TraceEvent:
    """The single collected trace event of `event_type`."""
    crewai_event_bus.flush()
    matches = [e for e in listener.batch_manager.event_buffer if e.type == event_type]
    assert len(matches) == 1, (
        f"expected one {event_type!r} trace event, got {len(matches)}: "
        f"{[e.type for e in listener.batch_manager.event_buffer]}"
    )
    return matches[0]


def test_human_feedback_requested_is_collected_whole(listener) -> None:
    crewai_event_bus.emit(
        None,
        HumanFeedbackRequestedEvent(
            flow_name="review_flow",
            method_name="draft",
            output="the draft shown to the reviewer",
            message="Approve this draft?",
            emit=["approved", "rejected"],
            request_id="req-1",
        ),
    )

    data = _collected(listener, "human_feedback_requested").event_data

    assert data["flow_name"] == "review_flow"
    assert data["method_name"] == "draft"
    assert data["output"] == "the draft shown to the reviewer"
    assert data["message"] == "Approve this draft?"
    assert data["emit"] == ["approved", "rejected"]
    assert data["request_id"] == "req-1"


def test_human_feedback_received_is_collected_whole(listener) -> None:
    crewai_event_bus.emit(
        None,
        HumanFeedbackReceivedEvent(
            flow_name="review_flow",
            method_name="draft",
            feedback="Looks good, ship it.",
            outcome="approved",
            request_id="req-1",
        ),
    )

    data = _collected(listener, "human_feedback_received").event_data

    assert data["flow_name"] == "review_flow"
    assert data["method_name"] == "draft"
    assert data["feedback"] == "Looks good, ship it."
    assert data["outcome"] == "approved"
    assert data["request_id"] == "req-1"


def test_method_execution_paused_is_collected_whole(listener) -> None:
    crewai_event_bus.emit(
        None,
        MethodExecutionPausedEvent(
            flow_name="review_flow",
            method_name="draft",
            state={"draft": "v1"},
            flow_id="flow-1",
            message="Approve this draft?",
            emit=["approved", "rejected"],
        ),
    )

    data = _collected(listener, "method_execution_paused").event_data

    assert data["flow_name"] == "review_flow"
    assert data["method_name"] == "draft"
    assert data["flow_id"] == "flow-1"
    assert data["state"] == {"draft": "v1"}
    assert data["message"] == "Approve this draft?"
    assert data["emit"] == ["approved", "rejected"]


def test_flow_paused_is_collected_whole(listener) -> None:
    crewai_event_bus.emit(
        None,
        FlowPausedEvent(
            flow_name="review_flow",
            flow_id="flow-1",
            method_name="draft",
            state={"draft": "v1"},
            message="Approve this draft?",
            emit=["approved", "rejected"],
        ),
    )

    data = _collected(listener, "flow_paused").event_data

    assert data["flow_name"] == "review_flow"
    assert data["flow_id"] == "flow-1"
    assert data["method_name"] == "draft"
    assert data["state"] == {"draft": "v1"}
    assert data["message"] == "Approve this draft?"
    assert data["emit"] == ["approved", "rejected"]
