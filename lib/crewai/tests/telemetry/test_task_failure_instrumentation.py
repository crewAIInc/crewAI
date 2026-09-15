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

from contextlib import contextmanager
from unittest.mock import Mock, patch
import sys
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


@contextmanager
def captured_task_failures():
    """Every TaskFailedEvent the producer *emits*, captured at the emit boundary.

    Patches ``crewai_event_bus.emit`` rather than subscribing a handler to it. What these
    tests are about is what the producer in ``task.py`` constructs, and a subscriber makes
    that claim depend on the bus choosing to dispatch -- which it does not always do.
    ``task_failed`` is an "ending" event, and with an empty scope stack (there is no real
    kickoff here) dispatch is conditional on event-context state that other tests in the
    same worker process can leave behind. That is not a hypothetical: subscribing passed
    this file in isolation and every randomized local run, then failed in CI inside a
    621-test shard with zero events captured.

    The emitted return value is unused by both producers, so returning None is faithful.
    """
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.task_events import TaskFailedEvent

    captured = []

    def _record(_source, event):
        if isinstance(event, TaskFailedEvent):
            captured.append(event)
        return None

    with patch.object(crewai_event_bus, "emit", side_effect=_record):
        yield captured


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

    with captured_task_failures() as captured:
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

    # aexecute_task, not execute_task: the async producer calls a different agent
    # method, which is precisely why it can regress independently of the sync one.
    with captured_task_failures() as captured:
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

    with captured_task_failures() as captured:
        with pytest.raises(Exception, match="has no agent assigned"):
            task._execute_core(None, None, None)

    assert len(captured) == 1
    assert captured[0].error_type is Exception


def test_the_event_stays_json_serializable_with_a_class_valued_error_type():
    """A class is not a JSON type, so the field needs a serializer or the event breaks.

    Without one, `model_dump(mode="json")` raises PydanticSerializationError for the
    *whole* event, not just this field. Two real consumers depend on it: the checkpoint
    listener dumps every event through EventRecord, and the tracing listener JSON-POSTs
    events to AMP. So a task failure would take out checkpointing entirely.
    """
    from crewai.events.types.task_events import TaskFailedEvent

    event = TaskFailedEvent(error="boom", error_type=ValueError, task=None)

    assert event.model_dump(mode="json")["error_type"] == "builtins:ValueError"
    assert "ValueError" in event.model_dump_json()


def test_python_mode_keeps_the_live_class_for_telemetry():
    """`when_used="json"` is load-bearing and must not be widened to every mode.

    `event_listener` hands `event.error_type` to `Telemetry.task_failed`, which runs it
    through `_safe_error_type` -- that requires the class object, not its name. If the
    serializer applied in python mode too, telemetry would receive a string, silently
    fail `isinstance(error_type, type)`, and record nothing.
    """
    from crewai.events.types.task_events import TaskFailedEvent

    event = TaskFailedEvent(error="boom", error_type=ValueError, task=None)

    assert event.model_dump(mode="python")["error_type"] is ValueError
    assert event.error_type is ValueError


def test_an_absent_error_type_serializes_as_null_not_as_a_string():
    from crewai.events.types.task_events import TaskFailedEvent

    event = TaskFailedEvent(error="boom", task=None)

    assert event.model_dump(mode="json")["error_type"] is None


def test_a_dumped_event_restores_as_itself_and_keeps_both_error_fields():
    """The JSON dump must round-trip, or restoring a checkpoint loses the failure.

    `_resolve_event` in state/event_record.py wraps `cls.model_validate` in a bare
    `except Exception` and falls back to `BaseEvent`. So a class-name string the field
    would not accept does not raise -- it silently degrades the whole event, taking
    `error` with it even though `error` is a plain string that would have survived.
    That is worse than the raise this serializer was added to prevent.
    """
    from crewai.events.types.task_events import TaskFailedEvent
    from crewai.state.event_record import EventRecord

    event = TaskFailedEvent(error="boom", error_type=ValueError, task=None)
    record = EventRecord()
    record.add(event)

    restored = EventRecord.model_validate(record.model_dump(mode="json")).nodes[
        event.event_id
    ].event

    assert type(restored).__name__ == "TaskFailedEvent", (
        "the event degraded to a bare BaseEvent, so the whole failure was lost"
    )
    assert restored.error == "boom"
    assert restored.error_type is ValueError


def test_resolving_a_name_cannot_smuggle_in_a_message():
    """Accepting the serialized name must not reopen the hole the class type closes.

    Resolution is against real exception classes only, so an identifier-shaped message
    resolves to nothing and is then rejected by the field's own type.
    """
    import pydantic
    from crewai.events.types.task_events import TaskFailedEvent

    for not_an_exception in ("secret_token", "sk_live_1234", "dict", "os"):
        with pytest.raises(pydantic.ValidationError, match="subclass of BaseException"):
            TaskFailedEvent(error="boom", error_type=not_an_exception, task=None)


def test_a_non_builtin_exception_class_also_round_trips():
    """Most failures here are not builtins -- provider and pydantic errors dominate."""
    from crewai.events.types.task_events import TaskFailedEvent

    event = TaskFailedEvent(error="boom", error_type=_ProducerFailure, task=None)
    dumped = event.model_dump(mode="json")

    assert dumped["error_type"] == f"{__name__}:_ProducerFailure"
    assert TaskFailedEvent.model_validate(dumped).error_type is _ProducerFailure


def test_two_exception_classes_sharing_a_name_cannot_be_confused():
    """A same-named class must never be substituted for the one that was serialized.

    Regression for the review finding on this PR: resolving by bare ``__name__`` through
    ``BaseException.__subclasses__()`` returned whichever same-named class the walk
    reached first, so an event serialized with one ``DupErr`` restored as a different
    ``DupErr``. Both are created here exactly as the review reproduced it.
    """
    from crewai.events.types.task_events import TaskFailedEvent

    first = type("DupErr", (Exception,), {})
    second = type("DupErr", (Exception,), {})
    assert first is not second and first.__name__ == second.__name__

    # Reachable by attribute lookup, so resolution can succeed for the right one.
    first.__module__ = __name__
    first.__qualname__ = "_dup_first"
    second.__module__ = __name__
    second.__qualname__ = "_dup_second"
    globals()["_dup_first"] = first
    globals()["_dup_second"] = second
    try:
        dumped = TaskFailedEvent(
            error="boom", error_type=first, task=None
        ).model_dump(mode="json")

        restored = TaskFailedEvent.model_validate(dumped).error_type

        assert restored is first
        assert restored is not second, (
            "restored a different class that merely shares the name"
        )
    finally:
        del globals()["_dup_first"]
        del globals()["_dup_second"]


def test_a_class_the_running_process_cannot_reach_degrades_to_none_not_to_the_wrong_class():
    """An unresolvable identity must lose only ``error_type``, never bind to something else.

    A locally defined class is unreachable by attribute lookup -- its ``__qualname__``
    contains ``<locals>`` -- which is the same shape as a checkpoint written by a process
    that had a module this one does not. The event must still restore, keeping ``error``.
    """
    from crewai.events.types.task_events import TaskFailedEvent

    class LocallyDefined(Exception):
        pass

    dumped = TaskFailedEvent(
        error="boom", error_type=LocallyDefined, task=None
    ).model_dump(mode="json")
    assert "<locals>" in dumped["error_type"]

    restored = TaskFailedEvent.model_validate(dumped)

    assert restored.error_type is None
    assert restored.error == "boom", "the message must survive an unresolvable type"


def test_a_qualified_identity_cannot_name_a_non_exception_or_force_an_import():
    """Resolution is gated on being an exception class, and never imports.

    ``os:getcwd`` resolves to a function, and ``json:JSONDecodeError`` is an exception
    but reachable only if ``json`` is already loaded -- neither may be recorded, and no
    module absent from ``sys.modules`` may be imported to satisfy a lookup.
    """
    from crewai.events.types.task_events import TaskFailedEvent

    unloaded = "xml.dom.minidom"
    # Restored in `finally`: leaving sys.modules mutated would make this suite
    # order-dependent, and the runner shuffles tests.
    previous = sys.modules.pop(unloaded, None)
    try:
        for identity in (
            "os:getcwd",
            "os:path",
            f"{unloaded}:Node",
            "nonexistent_mod:Boom",
        ):
            event = TaskFailedEvent(error="boom", error_type=identity, task=None)
            assert event.error_type is None, identity

        assert unloaded not in sys.modules, "resolution imported a module"
    finally:
        if previous is not None:
            sys.modules[unloaded] = previous


def test_a_message_containing_a_colon_is_never_stored():
    """Messages routinely contain colons, and must not be mistaken for an identity."""
    from crewai.events.types.task_events import TaskFailedEvent

    for message in (
        "AuthenticationError: invalid api key sk_live_1234",
        "connection refused: 10.0.0.1:5432",
        "secret_token:hunter2",
    ):
        event = TaskFailedEvent(error=message, error_type=message, task=None)
        assert event.error_type is None, message


def test_resolution_does_not_trigger_a_lazy_module_getattr():
    """A PEP 562 ``__getattr__`` must never run: it is an import in disguise.

    Review finding on this PR. ``getattr(module, name)`` invokes a module-level
    ``__getattr__`` on any miss, and such a hook typically calls
    ``importlib.import_module`` -- ``crewai.events`` itself is one. So an attribute walk
    would import a submodule named in the serialized string and run its top-level code,
    which is exactly the import primitive this resolver must not be. Reading ``__dict__``
    consults no hook.
    """
    import types

    from crewai.events.types.task_events import TaskFailedEvent

    invoked: list[str] = []
    lazy = types.ModuleType("lazy_probe_pkg")

    def _lazy_getattr(name: str) -> object:
        invoked.append(name)  # stands in for importlib.import_module
        raise AttributeError(name)

    lazy.__getattr__ = _lazy_getattr  # type: ignore[attr-defined]
    sys.modules["lazy_probe_pkg"] = lazy
    try:
        event = TaskFailedEvent(
            error="boom", error_type="lazy_probe_pkg:Something", task=None
        )
    finally:
        del sys.modules["lazy_probe_pkg"]

    assert invoked == [], f"resolution ran the module's __getattr__: {invoked}"
    assert event.error_type is None


def test_a_raising_lazy_hook_cannot_degrade_the_event():
    """A hook raising anything but AttributeError must not escape validation.

    ``getattr(obj, name, None)`` only swallows ``AttributeError``. Anything else would
    propagate out of the validator, and ``_resolve_event`` would catch it and degrade the
    whole event to a bare ``BaseEvent`` -- dropping ``error`` again, which is the
    regression this resolver was rewritten to prevent.
    """
    import types

    from crewai.events.types.task_events import TaskFailedEvent

    exploding = types.ModuleType("exploding_probe_pkg")

    def _exploding_getattr(name: str) -> object:
        raise RuntimeError("lazy loader exploded")

    exploding.__getattr__ = _exploding_getattr  # type: ignore[attr-defined]
    sys.modules["exploding_probe_pkg"] = exploding
    try:
        event = TaskFailedEvent(
            error="boom", error_type="exploding_probe_pkg:Something", task=None
        )
    finally:
        del sys.modules["exploding_probe_pkg"]

    assert event.error_type is None
    assert event.error == "boom", "the message must survive a hostile lazy hook"
