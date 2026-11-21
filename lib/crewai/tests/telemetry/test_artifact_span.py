"""A saved artifact shows up as a span, with who saved it."""

from __future__ import annotations

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.artifact_events import ArtifactCreatedEvent
from crewai.telemetry.tracing import handlers
from crewai.telemetry.tracing.context import TelemetryExecutionContext
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


@pytest.fixture(autouse=True)
def enable_otel_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    # The suite runs with OTEL_SDK_DISABLED, which would make spans non-recording.
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TRACKING", raising=False)


class _Providers:
    def __init__(self, tracer) -> None:
        self._tracer = tracer

    def get_tracer(self, name: str | None = None):
        return self._tracer

    def emit_log(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
def pipeline():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    ctx = TelemetryExecutionContext(kickoff_id="kickoff", automation_name="test", tracer=tracer)
    yield _Providers(tracer), ctx, exporter


def _event() -> ArtifactCreatedEvent:
    return ArtifactCreatedEvent(
        id="0b9e6c1e-0000-4000-8000-000000000000",
        identifier="q3-report",
        version=2,
        title="Q3 report",
        kind="html",
        content_type="text/html",
        size=12,
        url="file:///artifacts/v2.html",
        task_id="task-1",
        agent_role="Analyst",
    )


def test_artifact_created_becomes_a_span(pipeline):
    providers, ctx, exporter = pipeline

    handlers.handle_artifact_created(providers, ctx, None, _event())

    [span] = [s for s in exporter.get_finished_spans() if s.name == "save artifact"]
    attrs = dict(span.attributes)
    assert attrs["crewai.artifact.identifier"] == "q3-report"
    assert attrs["crewai.artifact.version"] == 2
    assert attrs["crewai.artifact.kind"] == "html"
    assert attrs["crewai.artifact.url"] == "file:///artifacts/v2.html"
    assert attrs["crewai.agent.role"] == "Analyst"
    assert attrs["crewai.task.id"] == "task-1"


def test_artifact_created_is_registered(pipeline):
    providers, ctx, _ = pipeline
    with crewai_event_bus.scoped_handlers():
        registered = handlers.register_handlers(providers, ctx, crewai_event_bus)
    assert ArtifactCreatedEvent in [event_type for event_type, _ in registered]
