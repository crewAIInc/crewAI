"""Process exit must not wait on telemetry retries.

``BatchSpanProcessor.force_flush`` ignores its timeout, so with the collector
unreachable the exit hook used to block for the exporter's whole retry budget.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import logging
import time
from typing import cast
from unittest.mock import MagicMock, patch

import crewai_core.telemetry as telemetry_module
from crewai_core.telemetry import SafeOTLPSpanExporter, Telemetry
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import pytest
import requests


OTLP_LOGGER = OTLPSpanExporter.__module__
ENDPOINT = "http://127.0.0.1:9/v1/traces"


def _otlp_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == OTLP_LOGGER]


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
        if telemetry.ready:
            telemetry.provider.shutdown()
        Telemetry._instance = None


def test_exit_hook_stops_waiting_at_the_flush_deadline(
    live_telemetry: Telemetry,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(telemetry_module, "FINAL_FLUSH_SECONDS", 1)
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

    # The exporter's own retry budget is 30s; unbounded, this takes 15s or more.
    assert elapsed < 5
    assert live_telemetry.ready is False
    assert _otlp_messages(caplog) == []


def test_exit_hook_returns_as_soon_as_the_flush_succeeds(
    live_telemetry: Telemetry, caplog: pytest.LogCaptureFixture
) -> None:
    live_telemetry.provider.get_tracer("test").start_span("probe").end()

    with (
        patch("requests.Session.post", return_value=MagicMock(ok=True)) as post,
        caplog.at_level(logging.DEBUG),
    ):
        started = time.monotonic()
        live_telemetry._shutdown()
        elapsed = time.monotonic() - started

    assert post.call_count == 1
    assert elapsed < 5  # not the 10s deadline
    assert _otlp_messages(caplog) == []


def test_repeated_exporter_shutdown_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    exporter = SafeOTLPSpanExporter(endpoint=ENDPOINT, timeout=1)

    with caplog.at_level(logging.DEBUG):
        exporter.shutdown()
        exporter.shutdown()

    assert _otlp_messages(caplog) == []


def test_repeated_plain_exporter_shutdown_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Control for the test above: the base class does warn on the repeat call."""
    exporter = OTLPSpanExporter(endpoint=ENDPOINT, timeout=1)
    # OTLPSpanExporter.shutdown is unannotated upstream.
    plain_shutdown = cast(Callable[[], None], exporter.shutdown)

    with caplog.at_level(logging.DEBUG):
        plain_shutdown()
        plain_shutdown()

    assert "Exporter already shutdown, ignoring call" in _otlp_messages(caplog)
