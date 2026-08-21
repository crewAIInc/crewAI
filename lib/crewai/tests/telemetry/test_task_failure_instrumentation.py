"""Failed task executions must be recorded as failures, not as successes.

`close_span()` sets `StatusCode.OK` unconditionally, and `TaskFailedEvent` was routed
to `Telemetry.task_ended`, which calls it. Every failed task was therefore exported as
OK, which is why `error_count` is not merely low downstream but exactly zero for every
month on record -- 240.0M task executions across 13 months in
`crew_task_executions_daily_target`, `error_count = 0` in all of them.

Scope is the task path only. The crew half of the original change (closed PR #6781) is
deliberately absent: `crew_execution_span()` returns `None` unless `share_crew=True`, so
`crew._execution_span` is `None` for nearly every user and a crew-failure handler would
have exited immediately for the default population.
"""

from unittest.mock import Mock, patch
import threading

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from crewai.telemetry.utils import close_span, close_span_with_error


@pytest.fixture(autouse=True)
def enable_otel_sdk(monkeypatch):
    """Ensure the OTel SDK is active for these tests.

    The suite otherwise runs with OTEL_SDK_DISABLED=true, which makes TracerProvider
    hand out non-recording spans that are never exported -- every assertion here would
    pass vacuously. Set via monkeypatch rather than relying on the root conftest, which
    pops the variable on teardown and would leave only the first test in a session
    running against a disabled SDK.
    """
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("CREWAI_DISABLE_TRACKING", raising=False)


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    # yield rather than return: the generator frame keeps `provider` alive for the
    # duration of the test. If it is collected, its processor shuts down and spans
    # are silently lost.
    yield exp, provider.get_tracer("test")


def test_close_span_with_error_sets_error_status(exporter):
    exp, tracer = exporter

    close_span_with_error(tracer.start_span("Task Execution"), "ValidationError")

    span = exp.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error_type"] == "ValidationError"


def test_successful_and_failed_spans_are_distinguishable(exporter):
    """The whole point: a downstream count of failures must be possible at all."""
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

    Error messages routinely contain prompts, model output and credentials. Passing one
    where an exception class name belongs must record nothing at all -- the span is
    still ERROR, but no attribute is written.
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


def test_task_failed_closes_span_with_error(exporter):
    from crewai.telemetry.telemetry import Telemetry

    exp, tracer = exporter
    telemetry = Telemetry()
    telemetry.ready = True

    telemetry.task_failed(
        tracer.start_span("Task Execution"), Mock(fingerprint=None), ValueError
    )

    finished = exp.get_finished_spans()[0]
    assert finished.status.status_code is StatusCode.ERROR
    assert finished.attributes["error_type"] == "ValueError"


def test_task_failed_event_carries_the_class_and_not_the_message():
    from crewai.events.types.task_events import TaskFailedEvent

    try:
        raise TimeoutError("request to gpt-4o timed out after 60s")
    except TimeoutError as e:
        event = TaskFailedEvent(error=str(e), error_type=type(e), task=None)

    assert event.error_type is TimeoutError
    assert "gpt-4o" not in str(event.error_type)


def test_a_message_cannot_be_passed_as_the_error_type_at_all():
    """The field takes the exception *class*, so a message is rejected structurally.

    An identifier check on a name is not sufficient on its own -- a one-word message
    such as "secret_token" is a valid identifier -- which is exactly why
    ``Telemetry._safe_error_type`` takes a class rather than a string. Making the event
    field a class means pydantic refuses a message before any of our code runs.
    """
    import pydantic
    from crewai.events.types.task_events import TaskFailedEvent

    with pytest.raises(pydantic.ValidationError, match="subclass of BaseException"):
        TaskFailedEvent(error="boom", error_type="secret_token", task=None)


def test_task_failed_discards_anything_that_is_not_an_exception_class(exporter):
    """A non-exception reaching task_failed records nothing, and still closes as ERROR."""
    from crewai.telemetry.telemetry import Telemetry

    exp, tracer = exporter
    telemetry = Telemetry()
    telemetry.ready = True

    telemetry.task_failed(
        tracer.start_span("Task Execution"),
        Mock(fingerprint=None),
        "secret_token",  # type: ignore[arg-type]  - deliberately wrong, as a caller might
    )

    span = exp.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.ERROR
    assert "error_type" not in (span.attributes or {}), (
        "an identifier-shaped message must not be recorded just because it parses"
    )


def test_error_type_defaults_to_none_for_backwards_compatibility():
    """Existing callers that construct the event without error_type must keep working."""
    from crewai.events.types.task_events import TaskFailedEvent

    assert TaskFailedEvent(error="boom", task=None).error_type is None


@pytest.fixture
def listener_with_a_recording_telemetry():
    """A real EventListener whose telemetry closes spans for real, but nothing else.

    The handler under test is reached through the real event bus, so the routing is
    genuinely exercised. Only ``Telemetry`` is substituted, and its two relevant methods
    keep their real span-closing behaviour via ``close_span``/``close_span_with_error``.

    Substituted rather than used live because ``_safe_telemetry_operation`` swallows every
    exception and returns None: with the real Telemetry, anything that makes it bail --
    a readiness flag, an env var, a provider already installed by another test in the
    session -- produces an unclosed span and a failure that looks exactly like the
    regression this test exists to catch. Verified: the failure moved between the two
    parameter cases depending only on which ran first in the process.

    Both singletons and the bus are reset on the way in and out, so no handler leaks
    into another test -- the suite runs in random order.
    """
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.event_listener import EventListener
    from crewai.telemetry import Telemetry

    def _reset():
        with crewai_event_bus._rwlock.w_locked():
            crewai_event_bus._sync_handlers.clear()
            crewai_event_bus._async_handlers.clear()
        Telemetry._instance = None
        EventListener._instance = None
        if hasattr(Telemetry, "_lock"):
            Telemetry._lock = threading.Lock()

    _reset()
    listener = EventListener()

    class _RecordingTelemetry:
        def __init__(self):
            self.calls = []

        def task_failed(self, span, task, error_type=None):
            self.calls.append("task_failed")
            # Mirrors the real method: the class is reduced to a safe name first.
            close_span_with_error(span, Telemetry._safe_error_type(error_type))

        def task_ended(self, span, task, crew):
            self.calls.append("task_ended")
            close_span(span)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    recording = _RecordingTelemetry()
    listener._telemetry = recording
    yield listener, crewai_event_bus, recording

    _reset()


def _handler_for(bus, event_type):
    """The handler `setup_listeners` actually registered for an event type.

    Called directly rather than through ``bus.emit``. ``emit`` carries its own
    event-context and runtime-scope state which is reset per test by an autouse fixture
    in the root conftest, and with an empty scope stack it does not reliably dispatch --
    observed here as the same assertion passing or failing depending only on which
    parametrized case ran first in the process. That machinery is not what this change
    touches. This still exercises the real closure built by ``setup_listeners``, bound to
    the real listener, so the routing under test is genuine; it just does not also depend
    on the emitter.
    """
    handlers = list(bus._sync_handlers.get(event_type, []))
    assert len(handlers) == 1, (
        f"expected exactly one registered handler for {event_type.__name__}, "
        f"got {[h.__qualname__ for h in handlers]}"
    )
    return handlers[0]


def _stub_task(crew):
    task = Mock()
    task.name = "some task"
    task.fingerprint = None
    task.agent = Mock(crew=crew, role="some role")
    task.output = None
    return task


@pytest.mark.parametrize(
    "crew", [Mock(share_crew=False), None], ids=["with-crew", "without-crew"]
)
def test_on_task_failed_closes_the_span_as_error_either_way(
    listener_with_a_recording_telemetry, exporter, crew
):
    """Wiring-level coverage, including the case that used to leak the span entirely.

    Two defects on one line. The handler routed to ``task_ended``, which closes the span
    as OK -- so a failed task was indistinguishable from a successful one downstream.
    And it only did so when ``source.agent.crew`` was present, so a task failing without
    one was popped from the span map and then never closed: never ended, never exported,
    invisible rather than merely mislabelled.

    This is the wiring test CodeRabbit asked for on the original PR and never got, which
    is why it is here and not only at the ``Telemetry.task_failed`` level.
    """
    from crewai.events.types.task_events import TaskFailedEvent

    listener, bus, recording = listener_with_a_recording_telemetry
    exp, tracer = exporter

    task = _stub_task(crew)
    listener.execution_spans[task] = tracer.start_span("Task Execution")

    handler = _handler_for(bus, TaskFailedEvent)
    handler(task, TaskFailedEvent(error="boom", error_type=ValueError, task=task))

    assert recording.calls == ["task_failed"], (
        "a failure must route to task_failed; task_ended would close the span as OK"
    )
    finished = exp.get_finished_spans()
    assert len(finished) == 1, "the span was never ended, so it would never be exported"
    assert finished[0].status.status_code is StatusCode.ERROR
    assert finished[0].attributes["error_type"] == "ValueError"
    assert task not in listener.execution_spans, "the span map leaked an entry"


def test_on_task_completed_still_routes_to_task_ended_and_closes_as_ok(
    listener_with_a_recording_telemetry, exporter
):
    """The success path shares this handler's span map and must be unaffected.

    ``on_task_failed`` and ``on_task_completed`` both pop from ``execution_spans``; only
    the failure route changed. If this regressed, every successful task would start
    reporting as an error and error_count would swing from zero to everything -- which is
    just as wrong and much harder to notice.
    """
    from crewai.events.types.task_events import TaskCompletedEvent
    from crewai.tasks.task_output import TaskOutput

    listener, bus, recording = listener_with_a_recording_telemetry
    exp, tracer = exporter

    task = _stub_task(Mock(share_crew=False))
    listener.execution_spans[task] = tracer.start_span("Task Execution")

    output = TaskOutput(description="some task", raw="done", agent="some role")
    handler = _handler_for(bus, TaskCompletedEvent)
    handler(task, TaskCompletedEvent(output=output, task=task))

    assert recording.calls == ["task_ended"]
    finished = exp.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].status.status_code is StatusCode.OK
    assert "error_type" not in (finished[0].attributes or {})


class _ProducerFailure(Exception):
    """Distinctive class, so the test cannot pass on a hardcoded or defaulted value."""


def _captured_task_failures():
    """Subscribe to TaskFailedEvent on the real bus and collect the events."""
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.task_events import TaskFailedEvent

    captured = []

    @crewai_event_bus.on(TaskFailedEvent)
    def _collect(_source, event):
        captured.append(event)

    return captured


@pytest.fixture
def failing_task():
    """A real Task whose agent raises, so the producer's except block is genuinely run."""
    from crewai import Agent, Task

    agent = Agent(
        role="tester",
        goal="fail",
        backstory="exists only to raise",
    )
    task = Task(description="a task that fails", expected_output="nothing", agent=agent)
    return task, agent


def test_sync_producer_puts_the_exception_class_on_the_event(failing_task):
    """`Task._execute_core` must populate error_type, not just `error`.

    The tests above construct TaskFailedEvent directly, so a regression in the producer
    itself -- the two emit sites in task.py -- would pass all of them. This drives the
    real execution path instead.
    """
    from crewai import Agent

    task, _agent = failing_task
    captured = _captured_task_failures()

    with patch.object(Agent, "execute_task", side_effect=_ProducerFailure("boom")):
        with pytest.raises(_ProducerFailure):
            task._execute_core(None, None, None)

    assert len(captured) == 1, "the producer must emit exactly one TaskFailedEvent"
    assert captured[0].error_type is _ProducerFailure
    assert captured[0].error == "boom"


@pytest.mark.asyncio
async def test_async_producer_puts_the_exception_class_on_the_event(failing_task):
    """The async producer is a separate emit site and regresses independently."""
    from crewai import Agent

    task, _agent = failing_task
    captured = _captured_task_failures()

    # aexecute_task, not execute_task: the async producer calls a different agent
    # method, which is precisely why it can regress independently of the sync one.
    with patch.object(Agent, "aexecute_task", side_effect=_ProducerFailure("boom")):
        with pytest.raises(_ProducerFailure):
            await task._aexecute_core(None, None, None)

    assert len(captured) == 1
    assert captured[0].error_type is _ProducerFailure
    assert captured[0].error == "boom"


def test_a_task_with_no_agent_still_reports_a_failure_type(failing_task):
    """The no-agent branch raises inside the same try, so it must report too."""
    from crewai import Task

    task = Task(description="orphan", expected_output="nothing")
    captured = _captured_task_failures()

    with pytest.raises(Exception, match="has no agent assigned"):
        task._execute_core(None, None, None)

    assert len(captured) == 1
    assert captured[0].error_type is not None
    assert issubclass(captured[0].error_type, BaseException)
