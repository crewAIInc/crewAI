from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import gc
from threading import Barrier
import weakref

from crewai.events import (
    HumanFeedbackReceivedEvent,
    HumanFeedbackRequestedEvent,
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    crewai_event_bus,
)
from crewai.events.types.llm_events import LLMCallType
from crewai.flow.flow import Flow, start
from crewai.telemetry.tracing.context import get_telemetry_context, get_trace_session
from crewai.telemetry.tracing.session import TraceSession, telemetry_session
from crewai.types.usage_metrics import UsageMetrics
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


@pytest.fixture(autouse=True)
def enable_sdk(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)


def complete_call(call_id):
    return LLMCallCompletedEvent(
        call_id=call_id, response="result", call_type=LLMCallType.LLM_CALL
    )


def test_parallel_sessions_keep_events_and_handlers_separate():
    ready = Barrier(2)
    observed = []

    def observer(source, event):
        observed.append(event.call_id)

    crewai_event_bus.on(LLMCallStartedEvent)(observer)

    def run(execution_id):
        exporter = InMemorySpanExporter()
        with telemetry_session(execution_id, "test", [exporter]):
            ready.wait(timeout=5)
            crewai_event_bus.emit(
                None, LLMCallStartedEvent(call_id=execution_id, messages="hello")
            )
            crewai_event_bus.emit(None, complete_call(execution_id))
        return exporter.get_finished_spans()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            executions = list(pool.map(run, ("first", "second")))
        for execution_id, spans in zip(("first", "second"), executions, strict=True):
            assert len(spans) == 1
            assert spans[0].attributes["crewai.execution_uuid"] == execution_id
            assert spans[0].attributes["crewai.llm.call_id"] == execution_id
        crewai_event_bus.emit(
            None, LLMCallStartedEvent(call_id="after-session", messages="hello")
        )
        crewai_event_bus.emit(None, complete_call("after-session"))
        assert crewai_event_bus.flush()
        assert sorted(observed) == ["after-session", "first", "second"]
    finally:
        crewai_event_bus.off(LLMCallStartedEvent, observer)


def test_event_spans_preserve_hierarchy_timestamps_and_status():
    exporter = InMemorySpanExporter()
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with telemetry_session("execution", "test", [exporter]):
        crewai_event_bus.emit(
            None,
            LLMCallStartedEvent(call_id="parent", messages="parent", timestamp=started),
        )
        crewai_event_bus.emit(
            None,
            LLMCallStartedEvent(
                call_id="child",
                messages="child",
                timestamp=started + timedelta(seconds=1),
            ),
        )
        child_end = complete_call("child")
        child_end.timestamp = started + timedelta(seconds=2)
        crewai_event_bus.emit(None, child_end)
        parent_end = complete_call("parent")
        parent_end.timestamp = started + timedelta(seconds=3)
        crewai_event_bus.emit(None, parent_end)
    by_call = {
        span.attributes["crewai.llm.call_id"]: span
        for span in exporter.get_finished_spans()
    }
    parent, child = by_call["parent"], by_call["child"]
    assert parent.parent is None
    assert child.parent.span_id == parent.context.span_id
    assert child.context.trace_id == parent.context.trace_id
    assert parent.end_time - parent.start_time == 3_000_000_000
    assert child.end_time - child.start_time == 1_000_000_000
    assert parent.status.status_code == child.status.status_code == trace.StatusCode.OK


def test_completed_spans_release_payloads_without_losing_parent_identity():
    session = TraceSession("execution")
    with session.activate():
        started = LLMCallStartedEvent(call_id="call", messages="large prompt" * 1000)
        crewai_event_bus.emit(None, started)
        assert crewai_event_bus.flush()
        span = session.context.active_spans[started.event_id]
        identity = span.get_span_context()
        reference = weakref.ref(span)
        del span
        crewai_event_bus.emit(None, complete_call("call"))
        assert crewai_event_bus.flush()
        gc.collect()
        assert reference() is None
        assert (
            session.context._span_refs[started.event_id].get_span_context() == identity
        )
        assert not hasattr(session.context._span_refs[started.event_id], "attributes")
    session.shutdown()


def test_human_feedback_spans_are_instant_children_of_the_flow():
    events = [
        HumanFeedbackRequestedEvent(
            flow_name="ReviewFlow",
            method_name="review",
            output="draft",
            message="Review",
        ),
        HumanFeedbackReceivedEvent(
            flow_name="ReviewFlow", method_name="review", feedback="approved"
        ),
    ]

    class ReviewFlow(Flow):
        @start()
        def review(self):
            for event in events:
                crewai_event_bus.emit(self, event)

    exporter = InMemorySpanExporter()
    with telemetry_session("execution", "test", [exporter]):
        ReviewFlow(tracing=False).kickoff()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    for name, event in zip(
        ("request human feedback", "receive human feedback"), events, strict=True
    ):
        span = spans[name]
        assert span.parent == spans["call method"].context
        assert span.context.trace_id == spans["execute flow"].context.trace_id
        assert (
            span.start_time == span.end_time == int(event.timestamp.timestamp() * 1e9)
        )
        assert span.status.status_code == trace.StatusCode.OK
        assert span.attributes["crewai.event_name"] == event.type


def test_session_owns_root_even_under_application_span():
    exporter = InMemorySpanExporter()
    application = trace.NonRecordingSpan(
        trace.SpanContext(
            trace_id=1234,
            span_id=5678,
            is_remote=False,
            trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
        )
    )
    with (
        trace.use_span(application),
        telemetry_session("execution", "test", [exporter]),
    ):
        crewai_event_bus.emit(
            None, LLMCallStartedEvent(call_id="call", messages="hello")
        )
        crewai_event_bus.emit(None, complete_call("call"))
    (span,) = exporter.get_finished_spans()
    assert span.parent is None
    assert span.context.trace_id != application.get_span_context().trace_id
    assert get_trace_session() is None and get_telemetry_context() is None


@pytest.mark.parametrize("fail", [False, True])
def test_tracing_preserves_flow_usage_metrics(fail):
    class UsageFlow(Flow):
        @start()
        def run(self):
            crewai_event_bus.emit(
                None, LLMCallStartedEvent(call_id="call", messages="hello")
            )
            completed = complete_call("call")
            completed.usage = {"prompt_tokens": 7, "completion_tokens": 3}
            crewai_event_bus.emit(None, completed)
            assert crewai_event_bus.flush()
            if fail:
                raise ValueError("flow failed after recording usage")
            return "done"

    exporter = InMemorySpanExporter()
    flow = UsageFlow(tracing=False)
    with telemetry_session("execution", "test", [exporter]):
        if fail:
            with pytest.raises(ValueError, match="flow failed"):
                flow.kickoff()
        else:
            assert flow.kickoff() == "done"

    assert flow.usage_metrics == UsageMetrics(
        total_tokens=10,
        prompt_tokens=7,
        completion_tokens=3,
        successful_requests=1,
    )
    flow_span = next(
        span for span in exporter.get_finished_spans() if span.name == "execute flow"
    )
    assert flow_span.status.status_code == (
        trace.StatusCode.ERROR if fail else trace.StatusCode.OK
    )


def test_shutdown_cleans_subscriptions_and_provider_after_event_drain_timeout(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    session = TraceSession("execution", [exporter])
    registered = tuple(session._registrations)
    monkeypatch.setattr(crewai_event_bus, "flush", lambda timeout: False)
    assert not session.shutdown()
    assert session._closed
    assert not session._registrations and not session.context._span_refs
    assert all(
        handler not in crewai_event_bus._sync_handlers.get(event_type, ())
        for event_type, handler in registered
    )
    assert session.shutdown()
