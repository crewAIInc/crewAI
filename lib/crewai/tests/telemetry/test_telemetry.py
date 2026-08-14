import ast
import inspect
import os
from pathlib import Path
import threading
from unittest.mock import Mock, patch

import pytest
from crewai import Agent, Crew, Task
from crewai.telemetry import Telemetry
from crewai_core.telemetry import Telemetry as CoreTelemetry
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture(autouse=True)
def cleanup_telemetry():
    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()
    yield
    Telemetry._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()


@pytest.mark.parametrize(
    "env_var,value,expected_ready",
    [
        ("OTEL_SDK_DISABLED", "true", False),
        ("OTEL_SDK_DISABLED", "TRUE", False),
        ("CREWAI_DISABLE_TELEMETRY", "true", False),
        ("CREWAI_DISABLE_TELEMETRY", "TRUE", False),
        ("OTEL_SDK_DISABLED", "false", True),
        ("CREWAI_DISABLE_TELEMETRY", "false", True),
    ],
)
def test_telemetry_environment_variables(env_var, value, expected_ready):
    """Test telemetry state with different environment variable configurations."""
    # Clear all telemetry-related env vars first, then set only the one being tested
    env_overrides = {
        "OTEL_SDK_DISABLED": "false",
        "CREWAI_DISABLE_TELEMETRY": "false",
        "CREWAI_DISABLE_TRACKING": "false",
        env_var: value,
    }
    with patch.dict(os.environ, env_overrides):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is expected_ready


def test_telemetry_enabled_by_default():
    """Test that telemetry is enabled by default."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is True


def test_set_tracer_never_installs_a_global_provider():
    """Telemetry must not hijack the process-wide TracerProvider.

    Installing it globally made every OTel-instrumented library in the host
    process export to CrewAI's collector, so the global provider must be left
    exactly as it was found whether or not an application installed one.
    """
    import opentelemetry.trace as ot

    with patch.dict(os.environ, {}, clear=True):
        before = ot.get_tracer_provider()
        telemetry = Telemetry()
        telemetry.set_tracer()
        after = ot.get_tracer_provider()

    assert after is before
    assert telemetry.trace_set is True


def test_flow_execution_span_records_crewai_version():
    tracer = Mock()
    span = Mock()
    tracer.start_span.return_value = span

    with (
        patch.dict(
            os.environ,
            {
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
        ),
        patch(
            "crewai.telemetry.telemetry.TracerProvider",
            return_value=Mock(get_tracer=Mock(return_value=tracer)),
        ),
        patch("crewai.telemetry.telemetry.version", return_value=_EMITTED_VERSION),
    ):
        telemetry = Telemetry()
        telemetry.flow_execution_span("ResearchFlow", ["start", "finish"])

    tracer.start_span.assert_called_once_with("Flow Execution")
    span.set_attribute.assert_any_call("crewai_version", "9.9.9")
    span.set_attribute.assert_any_call("flow_name", "ResearchFlow")


def test_flow_creation_span_records_crewai_version():
    tracer = Mock()
    span = Mock()
    tracer.start_span.return_value = span

    with (
        patch.dict(
            os.environ,
            {
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
        ),
        patch(
            "crewai.telemetry.telemetry.TracerProvider",
            return_value=Mock(get_tracer=Mock(return_value=tracer)),
        ),
        patch("crewai.telemetry.telemetry.version", return_value="9.9.9"),
    ):
        telemetry = Telemetry()
        # Flow creation also emits a once-per-process coding_agent feature span;
        # stub it so this test stays focused on the Flow Creation span.
        with patch.object(telemetry, "coding_agent_span"):
            telemetry.flow_creation_span("ResearchFlow")

    tracer.start_span.assert_called_once_with("Flow Creation")
    span.set_attribute.assert_any_call("crewai_version", "9.9.9")
    span.set_attribute.assert_any_call("flow_name", "ResearchFlow")


@patch("crewai.telemetry.telemetry.logger.error")
@patch(
    "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter.export",
    side_effect=Exception("Test exception"),
)
@pytest.mark.vcr()
def test_telemetry_fails_due_connect_timeout(export_mock, logger_mock):
    error = Exception("Test exception")
    export_mock.side_effect = error

    with patch.dict(
        os.environ, {"CREWAI_DISABLE_TELEMETRY": "false", "OTEL_SDK_DISABLED": "false"}
    ):
        telemetry = Telemetry()

        tracer = telemetry.provider.get_tracer(__name__)
        with tracer.start_as_current_span("test-span"):
            agent = Agent(
                role="agent",
                llm="gpt-4o-mini",
                goal="Just say hi",
                backstory="You are a helpful assistant that just says hi",
            )
            task = Task(
                description="Just say hi",
                expected_output="hi",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], name="TestCrew")
            crew.kickoff()

        telemetry.provider.force_flush()

    assert export_mock.called
    assert logger_mock.call_count == export_mock.call_count
    for call in logger_mock.call_args_list:
        assert call[0][0] == error


@pytest.mark.telemetry
def test_telemetry_singleton_pattern():
    """Test that Telemetry uses the singleton pattern correctly."""
    Telemetry._instance = None

    telemetry1 = Telemetry()
    telemetry2 = Telemetry()

    assert telemetry1 is telemetry2

    telemetry1.test_attribute = "test_value"
    assert hasattr(telemetry2, "test_attribute")
    assert telemetry2.test_attribute == "test_value"

    import threading

    instances = []

    def create_instance():
        instances.append(Telemetry())

    threads = [threading.Thread(target=create_instance) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(instance is telemetry1 for instance in instances)


def test_no_signal_handler_traceback_in_non_main_thread():
    """Signal handler registration should be silently skipped in non-main threads.

    Regression test for https://github.com/crewAIInc/crewAI/issues/4289
    """
    errors: list[Exception] = []
    mock_holder: dict = {}

    def init_in_thread():
        try:
            Telemetry._instance = None
            with (
                patch.dict(
                    os.environ,
                    {"CREWAI_DISABLE_TELEMETRY": "false", "OTEL_SDK_DISABLED": "false"},
                ),
                patch("crewai.telemetry.telemetry.TracerProvider"),
                patch("signal.signal") as mock_signal,
                patch("crewai.telemetry.telemetry.logger") as mock_logger,
            ):
                Telemetry()
                mock_holder["signal"] = mock_signal
                mock_holder["logger"] = mock_logger
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=init_in_thread)
    thread.start()
    thread.join()

    assert not errors, f"Unexpected error: {errors}"
    assert mock_holder, "Thread did not execute"
    mock_holder["signal"].assert_not_called()
    mock_holder["logger"].debug.assert_any_call(
        "Skipping signal handler registration: not running in main thread"
    )


def test_hook_dispatched_span_counts_point_usage():
    with (
        patch.dict(
            os.environ,
            {
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
        ),
        patch("crewai.telemetry.telemetry.TracerProvider"),
    ):
        telemetry = Telemetry()
        with patch.object(telemetry, "feature_usage_span") as feature_usage_span:
            telemetry.hook_dispatched_span("pre_tool_call", "proceeded")

    feature_usage_span.assert_called_once_with("hooks:pre_tool_call")


def test_hook_dispatched_span_counts_aborts():
    with (
        patch.dict(
            os.environ,
            {
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
        ),
        patch("crewai.telemetry.telemetry.TracerProvider"),
    ):
        telemetry = Telemetry()
        with patch.object(telemetry, "feature_usage_span") as feature_usage_span:
            telemetry.hook_dispatched_span("pre_tool_call", "aborted")

    feature_usage_span.assert_any_call("hooks:pre_tool_call")
    feature_usage_span.assert_any_call("hooks:aborted")
    assert feature_usage_span.call_count == 2


def test_event_listener_tracks_hook_dispatched_events():
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.event_listener import event_listener
    from crewai.events.types.hook_events import HookDispatchedEvent

    with (
        crewai_event_bus.scoped_handlers(),
        patch.object(
            event_listener._telemetry,
            "hook_dispatched_span",
        ) as hook_dispatched_span,
    ):
        event_listener.setup_listeners(crewai_event_bus)
        crewai_event_bus.emit(
            "test",
            HookDispatchedEvent(
                interception_point="pre_tool_call",
                outcome="aborted",
                hook_count=1,
                duration_ms=1.5,
            ),
        )
        crewai_event_bus.flush()

    hook_dispatched_span.assert_called_once_with(
        interception_point="pre_tool_call",
        outcome="aborted",
    )


# The version _emit injects. Assertions compare against this exact value, so a
# hard-coded literal in an emitter cannot satisfy them.
_EMITTED_VERSION = "9.9.9"


def _emit(method: str, *args, **kwargs):
    """Run one telemetry span method against a mocked tracer.

    The singleton is reset first: it caches the provider built on the very
    first construction, so without this only the earliest caller in a session
    would see the mocked tracer.
    """
    tracer = Mock()
    span = Mock()
    tracer.start_span.return_value = span
    Telemetry._instance = None

    with (
        patch.dict(
            os.environ,
            {
                "CREWAI_DISABLE_TELEMETRY": "false",
                "CREWAI_DISABLE_TRACKING": "false",
                "OTEL_SDK_DISABLED": "false",
            },
        ),
        patch(
            "crewai.telemetry.telemetry.TracerProvider",
            return_value=Mock(get_tracer=Mock(return_value=tracer)),
        ),
        patch("crewai.telemetry.telemetry.version", return_value="9.9.9"),
    ):
        getattr(Telemetry(), method)(*args, **kwargs)
    Telemetry._instance = None
    return tracer, span


@pytest.mark.parametrize(("resumed", "expected"), [(True, "true"), (False, "false")])
def test_resumed_is_recorded_as_a_string(resumed: bool, expected: str) -> None:
    """A boolean is encoded as the presence of a key, not as a value.

    ``false`` arrives as the key simply being absent, which is invisible in the
    schema and easy to extract wrongly - crew_memory reads 1 for 99.8% of crews
    for exactly that reason. A string leaves nothing to infer.
    """
    _tracer, span = _emit(
        "flow_execution_span", "ResearchFlow", ["start"], "user", resumed
    )

    span.set_attribute.assert_any_call("resumed", expected)
    for call in span.set_attribute.call_args_list:
        assert call.args[1] is not True and call.args[1] is not False


def test_flow_completed_records_duration_outcome_and_origin() -> None:
    _tracer, span = _emit("flow_completed_span", "ResearchFlow", 12.5, "failed", "user")

    span.set_attribute.assert_any_call("flow_name", "ResearchFlow")
    span.set_attribute.assert_any_call("duration_ms", 12.5)
    span.set_attribute.assert_any_call("outcome", "failed")
    span.set_attribute.assert_any_call("origin", "user")
    span.set_attribute.assert_any_call("conversational", "false")


@pytest.mark.parametrize(("flag", "expected"), [(True, "true"), (False, "false")])
def test_conversational_is_recorded_as_a_string(flag: bool, expected: str) -> None:
    """Same reason as resumed: a bool arrives as key presence, not a value."""
    _tracer, span = _emit(
        "flow_execution_span", "ResearchFlow", ["start"], "user", False, flag
    )

    span.set_attribute.assert_any_call("conversational", expected)


def test_paused_and_method_failed_record_flow_and_origin() -> None:
    for method in ("flow_paused_span", "flow_method_failed_span"):
        _tracer, span = _emit(method, "ResearchFlow", "internal")
        span.set_attribute.assert_any_call("flow_name", "ResearchFlow")
        span.set_attribute.assert_any_call("origin", "internal")


def _version_attr(span) -> str | None:
    """The crewai_version value recorded on a mocked span, if any."""
    for call in span.set_attribute.call_args_list:
        if call.args and call.args[0] == "crewai_version":
            return call.args[1]
    return None


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("flow_plotting_span", ("ResearchFlow", ["step_a", "step_b"])),
        ("deploy_signup_error_span", ()),
        ("start_deployment_span", ("dep-123",)),
        ("create_crew_deployment_span", ()),
        ("get_crew_logs_span", ("dep-123", "deployment")),
        ("remove_crew_span", ("dep-123",)),
        ("human_feedback_span", ("requested", False)),
    ],
)
def test_span_records_the_crewai_version(method: str, args: tuple) -> None:
    """Version-filtered queries silently drop any span kind missing this.

    Without it a release cannot be attributed for that span, so version-adoption
    and per-release regression analysis are blind to it.
    """
    _tracer, span = _emit(method, *args)

    # Exact equality with the value _emit injected: "looks like a version" would
    # also accept a hard-coded literal in the emitter.
    assert _version_attr(span) == _EMITTED_VERSION, (
        f"{method} did not record the value returned by version('crewai')"
    )


def test_task_spans_record_the_crewai_version() -> None:
    """Task Created and Task Execution are the highest-volume span kinds.

    They are emitted together by task_started, and both were missing the
    version - so every version-filtered task metric returned nothing.
    """
    agent = Agent(role="R", goal="G", backstory="B")
    task = Task(description="D", expected_output="E", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])

    tracer, span = _emit("task_started", crew, task)

    emitted = [c.args[0] for c in tracer.start_span.call_args_list]
    assert emitted == ["Task Created", "Task Execution"]

    # The harness hands the same mock back for both start_span calls, so the
    # attribute writes accumulate: one crewai_version per span emitted.
    versions = [
        c.args[1]
        for c in span.set_attribute.call_args_list
        if c.args and c.args[0] == "crewai_version"
    ]
    assert versions == [_EMITTED_VERSION, _EMITTED_VERSION], (
        f"expected one version per task span, got {versions}"
    )


def _calls(node: ast.AST, attr: str, key: str | None = None) -> int:
    """Count calls to ``.attr(...)`` beneath a node, optionally keyed on arg 2.

    Used to compare how many spans a method opens against how many of them it
    records ``crewai_version`` on.
    """
    total = 0
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if not isinstance(func, ast.Attribute) or func.attr != attr:
            continue
        if key is None:
            total += 1
        elif len(sub.args) >= 2:
            named = sub.args[1]
            if isinstance(named, ast.Constant) and named.value == key:
                total += 1
    return total


def test_every_span_records_the_crewai_version() -> None:
    """Regression guard for span kinds added later, in BOTH emitters.

    Enumerating the source rather than emitting all 32 spans: the point is to
    fail when someone adds a new span without the version, which a fixed list of
    behavioural cases cannot do.

    Counts rather than merely detects. A method that opens two spans and records
    the version on only one of them must fail - ``task_started`` is exactly that
    shape, so "the method mentions crewai_version somewhere" is not enough.
    """
    shortfalls: list[str] = []
    for cls in (Telemetry, CoreTelemetry):
        path = Path(inspect.getfile(cls))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # The nested closure is reached via its enclosing method, whose name
            # is the one a reader needs in the failure message.
            if not isinstance(node, ast.FunctionDef) or node.name == "_operation":
                continue
            spans = _calls(node, "start_span")
            if not spans:
                continue
            versions = _calls(node, "_add_attribute", "crewai_version")
            if versions < spans:
                shortfalls.append(
                    f"{path.name}::{node.name} "
                    f"({spans} span(s), {versions} version attribute(s))"
                )

    assert not shortfalls, (
        "these methods open more spans than they record crewai_version on: "
        + ", ".join(sorted(shortfalls))
    )
