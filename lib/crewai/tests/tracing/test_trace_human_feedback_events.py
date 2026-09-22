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
from datetime import datetime, timedelta, timezone
import os
from typing import Any
from unittest.mock import patch

from crewai.events.event_bus import crewai_event_bus
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.execution import clear_execution_uuid, set_execution_uuid
from crewai.flow.async_feedback import HumanFeedbackPending, PendingFeedbackContext
from crewai.flow.flow import Flow, listen, start
from crewai.flow.human_feedback import human_feedback
from crewai.flow.persistence.base import FlowPersistence
from crewai.telemetry.tracing.grants import GrantSpanExporter, TraceGrant, TraceGrantClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
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


def test_the_new_handlers_stay_idle_while_a_kickoff_owns_an_execution_uuid(listener) -> None:
    """The legacy collector must not run beside the OTEL session.

    Every handler in the listener registers through `_on`, which skips the
    event when an execution uuid is bound (the kickoff owns the new session and
    `telemetry/tracing/handlers.py` records these events there). The four gate
    and pause handlers, and the conversation handler, keep that gate: a run
    under an execution uuid collects none of them into a legacy batch.
    """
    token = set_execution_uuid("exec-owned-by-the-otel-session")
    try:
        crewai_event_bus.emit(
            object(),
            HumanFeedbackRequestedEvent(
                flow_name="review_flow",
                method_name="draft",
                output="the draft shown to the reviewer",
                message="Approve this draft?",
                emit=["approved", "rejected"],
                request_id="req-2",
            ),
        )
        crewai_event_bus.emit(
            object(),
            HumanFeedbackReceivedEvent(
                flow_name="review_flow",
                method_name="draft",
                feedback="Looks good, ship it.",
                outcome="approved",
                request_id="req-2",
            ),
        )
        crewai_event_bus.emit(
            object(),
            MethodExecutionPausedEvent(
                flow_name="review_flow",
                method_name="draft",
                state={"draft": "v1"},
                flow_id="flow-2",
                message="Approve this draft?",
                emit=["approved", "rejected"],
            ),
        )
        crewai_event_bus.emit(
            object(),
            FlowPausedEvent(
                flow_name="review_flow",
                flow_id="flow-2",
                method_name="draft",
                state={"draft": "v1"},
                message="Approve this draft?",
                emit=["approved", "rejected"],
            ),
        )
        crewai_event_bus.flush()
    finally:
        clear_execution_uuid(token)

    collected = [e.type for e in listener.batch_manager.event_buffer]
    assert not any(
        t in collected
        for t in (
            "human_feedback_requested",
            "human_feedback_received",
            "method_execution_paused",
            "flow_paused",
        )
    ), collected


# ---------------------------------------------------------------------------
# Under a tracing kickoff the OTEL session records the four events and the
# legacy collector stays idle — the two halves of the gate, on real flows.
# ---------------------------------------------------------------------------

LEGACY_TYPES = (
    "human_feedback_requested",
    "human_feedback_received",
    "method_execution_paused",
    "flow_paused",
)


@pytest.fixture
def session_recorders(monkeypatch) -> dict[str, InMemorySpanExporter]:
    """A tracing session that exports to memory, as in tests/telemetry/test_trace_lifecycle.py.

    No socket is opened: the grant is synthetic and the exporter records
    spans per execution uuid; anonymous consent is granted without a prompt.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_USER_PAT", raising=False)
    monkeypatch.delenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", raising=False)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setattr("crewai.telemetry.tracing.grants.get_auth_token", lambda: None)
    recorders: dict[str, InMemorySpanExporter] = {}

    def create(client: Any, execution_uuid: str) -> TraceGrant:
        return TraceGrant(
            token="synthetic-grant",
            collector_url="https://collector.invalid/v1/traces",
            execution_uuid=execution_uuid,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

    def exporter(grant: TraceGrant) -> InMemorySpanExporter:
        recorder = InMemorySpanExporter()
        recorders[grant.execution_uuid] = recorder
        return recorder

    monkeypatch.setattr(TraceGrantClient, "create", create)
    monkeypatch.setattr(GrantSpanExporter, "_exporter", staticmethod(exporter))
    return recorders


def _session_event_names(recorders: dict[str, InMemorySpanExporter]) -> set[str]:
    """Every `crewai.event_name` the session recorded, across executions."""
    return {
        str(span.attributes.get("crewai.event_name"))
        for recorder in recorders.values()
        for span in recorder.get_finished_spans()
        if span.attributes and span.attributes.get("crewai.event_name")
    }


def _legacy_types(listener: TraceCollectionListener) -> list[str]:
    crewai_event_bus.flush()
    return [e.type for e in listener.batch_manager.event_buffer if e.type in LEGACY_TYPES]


def test_a_gate_answered_in_place_is_recorded_by_the_session_not_the_legacy_batch(
    listener, session_recorders
) -> None:
    """A `@human_feedback` gate answered at the console, under a tracing kickoff.

    The kickoff owns an execution uuid, so the OTEL session records
    `human_feedback_requested` and `human_feedback_received` as spans and the
    legacy collector — gated by `_on` — collects neither.
    """

    class ReviewFlow(Flow):
        @start()
        @human_feedback(message="Approve this draft?")
        def draft(self) -> str:
            return "the draft"

        @listen(draft)
        def finish(self, result) -> str:
            return f"done: {result.feedback}"

    with (
        patch("builtins.input", return_value="looks good"),
        patch(
            "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
            return_value=True,
        ),
    ):
        result = ReviewFlow(tracing=True).kickoff()

    assert result == "done: looks good"
    recorded = _session_event_names(session_recorders)
    assert {"human_feedback_requested", "human_feedback_received"} <= recorded, recorded
    assert _legacy_types(listener) == []


def test_a_gate_that_pauses_the_flow_is_recorded_by_the_session_not_the_legacy_batch(
    listener, session_recorders
) -> None:
    """An async provider parks the flow: the session records the two pause
    events; the legacy collector, gated by `_on`, collects neither."""

    class MemoryPersistence(FlowPersistence):
        def init_db(self) -> None:
            pass

        def save_state(self, flow_uuid, method_name, state_data) -> None:
            pass

        def load_state(self, flow_uuid):
            return None

    class AsyncProvider:
        def request_feedback(self, context: PendingFeedbackContext, flow: Flow) -> str:
            raise HumanFeedbackPending(context=context)

    class PausingFlow(Flow):
        @start()
        @human_feedback(message="Approve this draft?", provider=AsyncProvider())
        def draft(self) -> str:
            return "the draft"

        @listen(draft)
        def finish(self, result) -> str:
            return f"done: {result.feedback}"

    with patch(
        "crewai.telemetry.tracing.ephemeral.prompt_user_for_trace_viewing",
        return_value=True,
    ):
        result = PausingFlow(persistence=MemoryPersistence(), tracing=True).kickoff()

    assert isinstance(result, HumanFeedbackPending)
    recorded = _session_event_names(session_recorders)
    assert {"method_execution_paused", "flow_paused"} <= recorded, recorded
    assert _legacy_types(listener) == []
