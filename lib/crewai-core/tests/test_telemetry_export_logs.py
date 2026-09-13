"""CrewAI's own telemetry exporter must fail silently.

When the collector is unreachable the OTLP exporter retries and logs a warning
per attempt plus a final error on the batch worker thread. That output reaches
the user's console and reads like a broken run, so it is dropped for our
exporter only. OTLP exporters the user configures keep their logs.
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
import threading
from unittest.mock import patch

from crewai_core.telemetry import SafeOTLPSpanExporter, Telemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import pytest
import requests


OTLP_LOGGER = OTLPSpanExporter.__module__
ENDPOINT = "http://127.0.0.1:9/v1/traces"
FINAL_ERROR = "Failed to export span batch due to timeout, max retries or shutdown."


def _finished_span() -> ReadableSpan:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    provider.get_tracer("test").start_span("probe").end()
    (span,) = memory.get_finished_spans()
    return span


def _otlp_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == OTLP_LOGGER]


def test_unreachable_collector_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    exporter = SafeOTLPSpanExporter(endpoint=ENDPOINT, timeout=1)

    with (
        patch(
            "requests.Session.post",
            side_effect=requests.exceptions.ConnectionError("collector down"),
        ),
        caplog.at_level(logging.DEBUG),
    ):
        result = exporter.export([_finished_span()])

    assert result is SpanExportResult.FAILURE
    assert _otlp_messages(caplog) == []


def test_user_otlp_exporters_keep_their_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Control for the test above: the same failure on a plain exporter does log."""
    SafeOTLPSpanExporter(endpoint=ENDPOINT, timeout=1)  # installs the filter
    exporter = OTLPSpanExporter(endpoint=ENDPOINT, timeout=1)

    with (
        patch(
            "requests.Session.post",
            side_effect=requests.exceptions.ConnectionError("collector down"),
        ),
        caplog.at_level(logging.DEBUG),
    ):
        result = exporter.export([_finished_span()])

    assert result is SpanExportResult.FAILURE
    assert FINAL_ERROR in _otlp_messages(caplog)


def test_filter_is_scoped_to_the_exporting_thread(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A user's exporter logging while ours is mid-retry on another thread is kept."""
    exporter = SafeOTLPSpanExporter(endpoint=ENDPOINT, timeout=1)
    entered = threading.Event()
    release = threading.Event()

    def blocked_post(*_args: object, **_kwargs: object) -> None:
        entered.set()
        release.wait(5)
        raise requests.exceptions.ConnectionError("collector down")

    with (
        patch("requests.Session.post", side_effect=blocked_post),
        caplog.at_level(logging.DEBUG),
    ):
        worker = threading.Thread(target=exporter.export, args=([_finished_span()],))
        worker.start()
        assert entered.wait(5)
        logging.getLogger(OTLP_LOGGER).warning("user exporter: collector down")
        release.set()
        worker.join(10)

    assert _otlp_messages(caplog) == ["user exporter: collector down"]


@pytest.fixture
def live_telemetry(monkeypatch: pytest.MonkeyPatch) -> Iterator[Telemetry]:
    """A fresh, enabled Telemetry with its real exporter and no lifecycle hooks."""
    monkeypatch.setattr(Telemetry, "_instance", None)
    monkeypatch.setattr(Telemetry, "_register_shutdown_handlers", lambda self: None)
    for var in (
        "CREWAI_DISABLE_TELEMETRY",
        "CREWAI_DISABLE_TRACKING",
        "OTEL_SDK_DISABLED",
    ):
        monkeypatch.setenv(var, "false")
    telemetry = Telemetry()
    try:
        yield telemetry
    finally:
        telemetry.provider.shutdown()
        Telemetry._instance = None


def test_telemetry_pipeline_is_silent_when_collector_rejects(
    live_telemetry: Telemetry, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch(
            "requests.Session.post",
            side_effect=requests.exceptions.HTTPError("400 Client Error"),
        ),
        caplog.at_level(logging.DEBUG),
    ):
        live_telemetry.provider.get_tracer("test").start_span("probe").end()
        assert live_telemetry.provider.force_flush(timeout_millis=10_000)

    assert _otlp_messages(caplog) == []
