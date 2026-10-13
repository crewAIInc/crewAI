"""The crewai Telemetry pipeline must not log when the collector is unreachable.

The filter lives in ``crewai_core``; this pins that the ``crewai`` package wires
the same exporter, since a duplicated exporter here used to log every failure.
"""

import logging
import time
from unittest.mock import patch

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import pytest
import requests

import crewai_core.telemetry as core_telemetry
from crewai.telemetry.telemetry import Telemetry


@pytest.fixture
def live_telemetry(monkeypatch):
    monkeypatch.setattr(Telemetry, "_instance", None)
    monkeypatch.setattr(Telemetry, "_register_shutdown_handlers", lambda self: None)
    for var in ("CREWAI_DISABLE_TELEMETRY", "CREWAI_DISABLE_TRACKING", "OTEL_SDK_DISABLED"):
        monkeypatch.setenv(var, "false")
    telemetry = Telemetry()
    try:
        yield telemetry
    finally:
        if telemetry.ready:
            telemetry.provider.shutdown()
        Telemetry._instance = None


def test_pipeline_is_silent_when_collector_rejects(live_telemetry, caplog):
    with (
        patch(
            "requests.Session.post",
            side_effect=requests.exceptions.HTTPError("400 Client Error"),
        ),
        caplog.at_level(logging.DEBUG),
    ):
        live_telemetry.provider.get_tracer("test").start_span("probe").end()
        assert live_telemetry.provider.force_flush(timeout_millis=10_000)

    otlp_records = [r for r in caplog.records if r.name == OTLPSpanExporter.__module__]
    assert otlp_records == []


def test_exit_hook_stops_waiting_at_the_flush_deadline(
    live_telemetry, monkeypatch, caplog
):
    monkeypatch.setattr(core_telemetry, "FINAL_FLUSH_SECONDS", 1)
    live_telemetry.provider.get_tracer("test").start_span("probe").end()

    with (
        patch(
            "requests.Session.post",
            side_effect=requests.exceptions.ConnectionError("collector down"),
        ),
        caplog.at_level(logging.DEBUG),
    ):
        started = time.monotonic()
        live_telemetry._shutdown()
        elapsed = time.monotonic() - started

    assert elapsed < 5  # the exporter's own retry budget is 30s
    assert live_telemetry.ready is False
    otlp_records = [r for r in caplog.records if r.name == OTLPSpanExporter.__module__]
    assert otlp_records == []
