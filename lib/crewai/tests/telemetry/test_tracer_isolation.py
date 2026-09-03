"""Telemetry must export our spans and only our spans.

Regression cover for the collector receiving third-party application traces:
``set_tracer()`` used to install CrewAI's ``TracerProvider`` as the global one,
so every OTel-instrumented library in the host process - HTTP servers, Redis
clients, ORMs - resolved ``trace.get_tracer()`` to our provider and shipped its
spans to CrewAI's endpoint.
"""

from typing import Any
from unittest.mock import patch

import opentelemetry.trace as ot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import pytest

from crewai.telemetry.constants import TRACER_NAME
from crewai.telemetry.telemetry import Telemetry


class _NullExporter:
    """Stands in for the OTLP exporter so no test attempts a real export."""

    def export(self, spans: Any) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@pytest.fixture
def telemetry_with_exporter(monkeypatch):
    """A fresh Telemetry whose provider exports into memory.

    Telemetry is a process-wide singleton that registers atexit and signal
    handlers on init, so the instance is replaced for the duration of the test
    and lifecycle registration is suppressed.
    """
    monkeypatch.setattr(Telemetry, "_instance", None)
    monkeypatch.setattr(Telemetry, "_register_shutdown_handlers", lambda self: None)

    # Set for the whole test: _is_telemetry_disabled() is re-read on every span
    # call, and the suite runs with OTEL_SDK_DISABLED set.
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "false")
    monkeypatch.setenv("CREWAI_DISABLE_TRACKING", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")

    # Patched before construction: __init__ wires the real OTLP exporter, which
    # would make every test here attempt a live export.
    monkeypatch.setattr(
        "crewai.telemetry.telemetry.SafeOTLPSpanExporter",
        lambda **_kwargs: _NullExporter(),
    )

    telemetry = Telemetry()

    exporter = InMemorySpanExporter()
    telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))

    try:
        yield telemetry, exporter
    finally:
        telemetry.provider.shutdown()
        Telemetry._instance = None


def test_third_party_spans_never_reach_our_exporter(telemetry_with_exporter):
    """A dependency instrumenting itself must not export to CrewAI."""
    telemetry, exporter = telemetry_with_exporter
    telemetry.set_tracer()

    ot.get_tracer("redis.client").start_span("XLEN").end()
    ot.get_tracer("opentelemetry.instrumentation.asgi").start_span(
        "GET /status http send"
    ).end()

    assert exporter.get_finished_spans() == ()


def test_our_own_spans_still_reach_our_exporter(telemetry_with_exporter):
    """The isolation must not cost us the telemetry we do want."""
    telemetry, exporter = telemetry_with_exporter
    telemetry.set_tracer()

    telemetry.feature_usage_span("cli_usage:view_traces")

    assert [span.name for span in exporter.get_finished_spans()] == ["Feature Usage"]


def test_our_spans_are_unaffected_by_an_application_provider(telemetry_with_exporter):
    """An app that installs its own provider must not divert our telemetry.

    Resolving our tracer globally meant that in an already-instrumented
    application our spans were created by the application's provider and went
    to its collector, so CrewAI received nothing at all from those processes.
    """
    telemetry, exporter = telemetry_with_exporter

    app_exporter = InMemorySpanExporter()
    app_provider = TracerProvider()
    app_provider.add_span_processor(SimpleSpanProcessor(app_exporter))

    with patch.object(ot, "get_tracer_provider", return_value=app_provider):
        telemetry.set_tracer()
        telemetry.feature_usage_span("cli_usage:deploy")

    assert [span.name for span in exporter.get_finished_spans()] == ["Feature Usage"]
    assert app_exporter.get_finished_spans() == ()


def test_set_tracer_is_idempotent(telemetry_with_exporter):
    """Repeated calls must not stack processors or duplicate exports."""
    telemetry, exporter = telemetry_with_exporter

    telemetry.set_tracer()
    telemetry.set_tracer()
    telemetry.set_tracer()

    telemetry.provider.get_tracer(TRACER_NAME).start_span("Crew Created").end()

    assert len(exporter.get_finished_spans()) == 1
