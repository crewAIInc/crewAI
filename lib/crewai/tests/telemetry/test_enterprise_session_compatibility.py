"""Behavioral compatibility at the hosted ``telemetry_session`` boundary."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    FlowFinishedEvent,
    FlowPausedEvent,
    FlowStartedEvent,
)
from crewai.execution import get_execution_uuid
from crewai.telemetry.tracing import telemetry_session
from crewai.telemetry.tracing.context import (
    get_execution_principal,
    get_telemetry_context,
    get_trace_session,
)
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


class HostProviders:
    """A host-owned provider with a custom resource, tracer, and log sink."""

    def __init__(self):
        self.exporter = InMemorySpanExporter()
        self.logs = []
        self.tracer_provider = TracerProvider(
            resource=Resource({"service.name": "hosted-automation"})
        )
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def get_tracer(self, name="crewai.enterprise"):
        return self.tracer_provider.get_tracer(name)

    def emit_log(self, body, **kwargs):
        self.logs.append((body, kwargs))

    def flush(self, timeout_millis=30000):
        return self.tracer_provider.force_flush(timeout_millis)

    def shutdown(self, timeout_millis=30000):
        self.tracer_provider.shutdown()
        return True


@pytest.fixture
def flow_source(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    return SimpleNamespace(
        name="HostedFlow",
        flow_id="flow-id",
        _methods={"review": None},
        suppress_flow_events=False,
    )


def test_host_provider_preserves_event_spans_and_logging_context(flow_source):
    providers = HostProviders()
    global_provider = trace.get_tracer_provider()
    principal = {"type": "user", "id": "operator-1"}
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    finished_at = started_at + timedelta(seconds=2)
    started = FlowStartedEvent(
        flow_name=flow_source.name, inputs={"topic": "test"}, timestamp=started_at
    )
    finished = FlowFinishedEvent(
        flow_name=flow_source.name,
        result="done",
        state={},
        timestamp=finished_at,
    )

    with crewai_event_bus.scoped_handlers():
        with telemetry_session(
            kickoff_id="kickoff-1",
            automation_name="hosted-automation",
            exporters=[],
            event_bus=crewai_event_bus,
            providers=providers,
            execution_id="execution-row-1",
            principal=principal,
            origin="schedule",
        ) as context:
            assert get_telemetry_context() is context
            assert get_execution_principal() == principal
            assert get_execution_uuid() == "kickoff-1"
            crewai_event_bus.emit(flow_source, started)
            crewai_event_bus.emit(flow_source, finished)

    (span,) = providers.exporter.get_finished_spans()
    assert span.name == "execute flow"
    assert span.instrumentation_scope.name == "crewai.enterprise"
    assert span.resource.attributes["service.name"] == "hosted-automation"
    assert span.start_time == int(started_at.timestamp() * 1_000_000_000)
    assert span.end_time == int(finished_at.timestamp() * 1_000_000_000)
    assert span.status.status_code == trace.StatusCode.OK
    assert span.attributes["crewai.execution_uuid"] == "kickoff-1"
    assert span.attributes["crewai.principal.id"] == "operator-1"
    assert span.attributes["crewai.execution.origin"] == "schedule"
    assert span.attributes["event_id"] == started.event_id
    assert {body for body, _ in providers.logs} == {
        "Flow started: HostedFlow",
        "Flow ended: HostedFlow",
    }
    for _, record in providers.logs:
        assert record["ctx"].execution_id == "execution-row-1"
        assert record["ctx"].principal == principal
    assert trace.get_tracer_provider() is global_provider
    assert get_telemetry_context() is None
    assert get_trace_session() is None


def test_host_can_supply_processors_and_a_logging_callback(flow_source):
    exporter = InMemorySpanExporter()
    logs = []

    def emit_log(body, **record):
        logs.append((f"host: {body}", record["ctx"].kickoff_id))

    with telemetry_session(
        kickoff_id="host-execution",
        automation_name=flow_source.name,
        processors=[SimpleSpanProcessor(exporter)],
        log_emitter=emit_log,
    ):
        crewai_event_bus.emit(flow_source, FlowStartedEvent(flow_name=flow_source.name))
        crewai_event_bus.emit(
            flow_source,
            FlowFinishedEvent(flow_name=flow_source.name, result="done", state={}),
        )

    (span,) = exporter.get_finished_spans()
    assert span.name == "execute flow"
    assert span.attributes["crewai.execution_uuid"] == "host-execution"
    assert set(logs) == {
        ("host: Flow started: HostedFlow", "host-execution"),
        ("host: Flow ended: HostedFlow", "host-execution"),
    }


def test_failed_session_keeps_existing_listeners_and_closes_orphans(flow_source):
    providers = HostProviders()
    received = []
    checkpoint_events = []
    execution_before = get_execution_uuid()

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(FlowStartedEvent)
        def host_listener(source, event):
            received.append(event.event_id)

        @crewai_event_bus.on(FlowStartedEvent)
        def checkpoint_listener(source, event):
            checkpoint_events.append(event.event_id)

        checkpoint_listener.__module__ = "crewai.state.checkpoint_listener"
        before = FlowStartedEvent(flow_name=flow_source.name)
        during = FlowStartedEvent(flow_name=flow_source.name)
        after = FlowStartedEvent(flow_name=flow_source.name)
        crewai_event_bus.emit(flow_source, before)
        assert crewai_event_bus.flush()

        def fail_session():
            with telemetry_session(
                kickoff_id="failed-kickoff",
                automation_name=flow_source.name,
                providers=providers,
                event_bus=crewai_event_bus,
            ):
                crewai_event_bus.emit(flow_source, during)
                raise RuntimeError("host execution failed")

        with pytest.raises(RuntimeError, match="host execution failed"):
            fail_session()

        crewai_event_bus.emit(flow_source, after)
        assert crewai_event_bus.flush()

    expected = [before.event_id, during.event_id, after.event_id]
    assert received == expected
    assert checkpoint_events == expected
    (span,) = providers.exporter.get_finished_spans()
    assert span.attributes["event_id"] == during.event_id
    assert span.status.status_code == trace.StatusCode.ERROR
    assert "orphaned" in span.status.description
    assert span.end_time is not None
    assert get_telemetry_context() is None
    assert get_trace_session() is None
    assert get_execution_uuid() == execution_before


def test_host_hitl_resume_links_segments_and_exposes_human_feedback(flow_source):
    first_providers = HostProviders()
    resumed_providers = HostProviders()

    with crewai_event_bus.scoped_handlers():
        with telemetry_session(
            kickoff_id="flow-execution",
            automation_name=flow_source.name,
            providers=first_providers,
            event_bus=crewai_event_bus,
        ) as paused_context:
            crewai_event_bus.emit(
                flow_source, FlowStartedEvent(flow_name=flow_source.name, inputs={})
            )
            crewai_event_bus.emit(
                flow_source,
                FlowPausedEvent(
                    flow_name=flow_source.name,
                    flow_id=flow_source.flow_id,
                    method_name="review",
                    state={},
                    message="Please approve",
                ),
            )

        (paused_span,) = first_providers.exporter.get_finished_spans()
        assert paused_span.attributes["crewai.event_name"] == "flow_paused"
        assert paused_span.status.status_code == trace.StatusCode.OK
        parent_context = paused_context.otel_resume_context
        assert parent_context == (
            paused_span.context.trace_id,
            paused_span.context.span_id,
        )

        with telemetry_session(
            kickoff_id="flow-execution",
            automation_name=flow_source.name,
            providers=resumed_providers,
            event_bus=crewai_event_bus,
            parent_otel_context=parent_context,
            resume_feedback="Approved by the reviewer",
            origin="hitl-resume",
        ):
            crewai_event_bus.emit(
                flow_source, FlowStartedEvent(flow_name=flow_source.name, inputs=None)
            )
            crewai_event_bus.emit(
                flow_source,
                FlowFinishedEvent(
                    flow_name=flow_source.name, result="approved", state={}
                ),
            )

    (resumed_span,) = resumed_providers.exporter.get_finished_spans()
    assert resumed_span.name == "execute flow"
    assert resumed_span.parent is None
    assert resumed_span.context.trace_id != paused_span.context.trace_id
    (link,) = resumed_span.links
    assert link.context.trace_id == paused_span.context.trace_id
    assert link.context.span_id == paused_span.context.span_id
    assert link.attributes["crewai.link.type"] == "follows_from"
    assert (
        "Approved by the reviewer" in resumed_span.attributes["gen_ai.input.messages"]
    )
    assert resumed_span.attributes["crewai.execution.origin"] == "hitl-resume"
    assert resumed_span.attributes["crewai.execution_uuid"] == "flow-execution"
