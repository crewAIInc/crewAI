"""Tests that failed executions are recorded as failures, not successes.

Regression coverage for telemetry that reported every task as OK, leaving
downstream error counts permanently at zero.
"""

from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from crewai.telemetry.utils import close_span, close_span_with_error


@pytest.fixture(autouse=True)
def enable_otel_sdk(monkeypatch):
    """Ensure the OTel SDK is active for these tests.

    The suite runs with OTEL_SDK_DISABLED=true, which makes TracerProvider hand
    out non-recording spans that are never exported. Set explicitly rather than
    relying on the root conftest teardown, which pops the variable and would
    otherwise leave only the first test in a session running against a
    disabled SDK.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TRACKING", raising=False)


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    # yield rather than return: the generator frame keeps `provider` alive for
    # the test. If it is collected, its processor shuts down and spans are lost.
    yield exp, provider.get_tracer("test")


def test_close_span_with_error_sets_error_status(exporter):
    exp, tracer = exporter

    close_span_with_error(tracer.start_span("Task Execution"), "ValidationError")

    span = exp.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error_type"] == "ValidationError"


def test_successful_and_failed_spans_are_distinguishable(exporter):
    """The whole point: a downstream count of failures must be possible."""
    exp, tracer = exporter

    close_span(tracer.start_span("Task Execution"))
    close_span_with_error(tracer.start_span("Task Execution"), "TimeoutError")
    close_span(tracer.start_span("Task Execution"))

    spans = exp.get_finished_spans()
    failed = [s for s in spans if s.status.status_code is StatusCode.ERROR]
    assert len(spans) == 3
    assert len(failed) == 1
    assert failed[0].attributes["error_type"] == "TimeoutError"


@pytest.mark.parametrize(
    "not_an_identifier",
    [
        "Rate limit exceeded for gpt-4o",
        "API key sk-live-1234 is invalid",
        "connection to db://user:pass@host failed",
        "",
        "  ",
        "429",
    ],
)
def test_error_message_can_never_be_recorded(exporter, not_an_identifier):
    """PII guard: only identifier-shaped values survive.

    Error messages routinely contain prompts, model output, and credentials.
    Passing one where an exception class name belongs must record nothing.
    """
    exp, tracer = exporter

    close_span_with_error(tracer.start_span("Task Execution"), not_an_identifier)

    span = exp.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert "error_type" not in (span.attributes or {})


def test_error_type_is_optional(exporter):
    exp, tracer = exporter

    close_span_with_error(tracer.start_span("Task Execution"))

    span = exp.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert "error_type" not in (span.attributes or {})


def test_real_exception_class_names_are_accepted(exporter):
    """Every builtin exception name is a valid identifier, so none are dropped."""
    exp, tracer = exporter

    for exc in (ValueError, TimeoutError, KeyError, RuntimeError, ConnectionError):
        close_span_with_error(tracer.start_span("Task Execution"), exc.__name__)

    recorded = [s.attributes["error_type"] for s in exp.get_finished_spans()]
    assert recorded == [
        "ValueError",
        "TimeoutError",
        "KeyError",
        "RuntimeError",
        "ConnectionError",
    ]


def test_task_failed_closes_span_with_error():
    from crewai.telemetry.telemetry import Telemetry

    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))

    telemetry = Telemetry()
    telemetry.ready = True
    span = provider.get_tracer("test").start_span("Task Execution")

    telemetry.task_failed(span, Mock(fingerprint=None), "ValueError")

    finished = exp.get_finished_spans()[0]
    assert finished.status.status_code is StatusCode.ERROR
    assert finished.attributes["error_type"] == "ValueError"


def test_crew_failed_closes_leaked_execution_span():
    """A crew that raises must not leave its span open and unexported."""
    from crewai.telemetry.telemetry import Telemetry

    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))

    telemetry = Telemetry()
    telemetry.ready = True

    crew = Mock()
    crew._execution_span = provider.get_tracer("test").start_span("Crew Execution")

    telemetry.crew_failed(crew, "RuntimeError")

    finished = exp.get_finished_spans()
    assert len(finished) == 1, "span was never ended - it would never be exported"
    assert finished[0].status.status_code is StatusCode.ERROR
    assert finished[0].attributes["error_type"] == "RuntimeError"
    assert crew._execution_span is None


def test_crew_failed_is_safe_when_no_span_exists():
    """share_crew=False crews have no execution span; this must not raise."""
    from crewai.telemetry.telemetry import Telemetry

    telemetry = Telemetry()
    telemetry.ready = True

    crew = Mock()
    crew._execution_span = None

    telemetry.crew_failed(crew, "RuntimeError")


def test_task_failed_event_carries_error_type():
    """The exception class must reach the event without the message."""
    from crewai.events.types.task_events import TaskFailedEvent

    try:
        raise TimeoutError("request to gpt-4o timed out after 60s")
    except TimeoutError as e:
        event = TaskFailedEvent(error=str(e), error_type=type(e).__name__, task=None)

    assert event.error_type == "TimeoutError"
    assert "gpt-4o" not in event.error_type


def test_crew_kickoff_failed_event_carries_error_type():
    from crewai.events.types.crew_events import CrewKickoffFailedEvent

    try:
        raise ValueError("bad input: {'api_key': 'sk-live-1234'}")
    except ValueError as e:
        event = CrewKickoffFailedEvent(
            error=str(e), error_type=type(e).__name__, crew_name="TestCrew"
        )

    assert event.error_type == "ValueError"
    assert "sk-live" not in event.error_type


def test_error_type_defaults_to_none_for_backwards_compatibility():
    """Existing callers that omit error_type must keep working."""
    from crewai.events.types.crew_events import CrewKickoffFailedEvent
    from crewai.events.types.task_events import TaskFailedEvent

    assert TaskFailedEvent(error="boom", task=None).error_type is None
    assert CrewKickoffFailedEvent(error="boom", crew_name="C").error_type is None
