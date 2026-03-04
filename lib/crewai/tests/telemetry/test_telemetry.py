import os
import threading
from unittest.mock import MagicMock, patch

import pytest
from crewai import Agent, Crew, Task
from crewai.telemetry import Telemetry
from opentelemetry import trace


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
        telemetry.set_tracer()

        tracer = trace.get_tracer(__name__)
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

        trace.get_tracer_provider().force_flush()

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


class TestCrewCreationTelemetryMemorySerialization:
    """Tests for issue #4703: telemetry fails with custom Memory instances.

    When crew.memory is a Memory object (not a bool), OpenTelemetry cannot
    serialize it as a span attribute. The fix converts non-primitive memory
    values to the class name string.
    """

    def test_crew_creation_telemetry_with_memory_true(self):
        """crew_memory attribute should be True (bool) when memory=True."""
        telemetry = Telemetry()
        telemetry.ready = True

        captured_attrs: dict[str, object] = {}
        original_add_attribute = telemetry._add_attribute

        def capture_add_attribute(span, key, value):
            captured_attrs[key] = value
            original_add_attribute(span, key, value)

        agent = Agent(
            role="researcher",
            goal="research",
            backstory="a researcher",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="do research",
            expected_output="results",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            memory=True,
        )

        with patch.object(telemetry, "_add_attribute", side_effect=capture_add_attribute):
            with patch.object(telemetry, "_safe_telemetry_operation") as mock_safe_op:
                # Call the inner _operation directly to test attribute logic
                mock_safe_op.side_effect = lambda op: op()
                with patch("crewai.telemetry.telemetry.trace") as mock_trace:
                    mock_span = MagicMock()
                    mock_trace.get_tracer.return_value.start_span.return_value = mock_span
                    telemetry.crew_creation(crew, None)

        assert captured_attrs.get("crew_memory") is True

    def test_crew_creation_telemetry_with_memory_false(self):
        """crew_memory attribute should be False (bool) when memory=False."""
        telemetry = Telemetry()
        telemetry.ready = True

        captured_attrs: dict[str, object] = {}
        original_add_attribute = telemetry._add_attribute

        def capture_add_attribute(span, key, value):
            captured_attrs[key] = value
            original_add_attribute(span, key, value)

        agent = Agent(
            role="researcher",
            goal="research",
            backstory="a researcher",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="do research",
            expected_output="results",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            memory=False,
        )

        with patch.object(telemetry, "_add_attribute", side_effect=capture_add_attribute):
            with patch.object(telemetry, "_safe_telemetry_operation") as mock_safe_op:
                mock_safe_op.side_effect = lambda op: op()
                with patch("crewai.telemetry.telemetry.trace") as mock_trace:
                    mock_span = MagicMock()
                    mock_trace.get_tracer.return_value.start_span.return_value = mock_span
                    telemetry.crew_creation(crew, None)

        assert captured_attrs.get("crew_memory") is False

    def test_crew_creation_telemetry_with_custom_memory_instance(self):
        """crew_memory attribute should be the class name when a Memory instance is passed.

        Regression test for https://github.com/crewAIInc/crewAI/issues/4703
        """
        telemetry = Telemetry()
        telemetry.ready = True

        captured_attrs: dict[str, object] = {}
        original_add_attribute = telemetry._add_attribute

        def capture_add_attribute(span, key, value):
            captured_attrs[key] = value
            original_add_attribute(span, key, value)

        # Create a mock Memory object to avoid needing real storage/LLM
        mock_memory = MagicMock()
        mock_memory.__class__.__name__ = "Memory"

        agent = Agent(
            role="researcher",
            goal="research",
            backstory="a researcher",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="do research",
            expected_output="results",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            memory=mock_memory,
        )

        with patch.object(telemetry, "_add_attribute", side_effect=capture_add_attribute):
            with patch.object(telemetry, "_safe_telemetry_operation") as mock_safe_op:
                mock_safe_op.side_effect = lambda op: op()
                with patch("crewai.telemetry.telemetry.trace") as mock_trace:
                    mock_span = MagicMock()
                    mock_trace.get_tracer.return_value.start_span.return_value = mock_span
                    telemetry.crew_creation(crew, None)

        # Should be the class name string, not the Memory object itself
        assert captured_attrs.get("crew_memory") == "Memory"
        assert isinstance(captured_attrs["crew_memory"], str)

    def test_crew_creation_telemetry_memory_value_is_otel_serializable(self):
        """The crew_memory value must always be a type OpenTelemetry can serialize.

        OpenTelemetry accepts: bool, str, bytes, int, float, or sequences thereof.
        """
        telemetry = Telemetry()
        telemetry.ready = True

        captured_attrs: dict[str, object] = {}
        original_add_attribute = telemetry._add_attribute

        def capture_add_attribute(span, key, value):
            captured_attrs[key] = value
            original_add_attribute(span, key, value)

        mock_memory = MagicMock()
        mock_memory.__class__.__name__ = "Memory"

        agent = Agent(
            role="researcher",
            goal="research",
            backstory="a researcher",
            llm="gpt-4o-mini",
        )
        task = Task(
            description="do research",
            expected_output="results",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            memory=mock_memory,
        )

        with patch.object(telemetry, "_add_attribute", side_effect=capture_add_attribute):
            with patch.object(telemetry, "_safe_telemetry_operation") as mock_safe_op:
                mock_safe_op.side_effect = lambda op: op()
                with patch("crewai.telemetry.telemetry.trace") as mock_trace:
                    mock_span = MagicMock()
                    mock_trace.get_tracer.return_value.start_span.return_value = mock_span
                    telemetry.crew_creation(crew, None)

        memory_val = captured_attrs.get("crew_memory")
        assert isinstance(memory_val, (bool, str, bytes, int, float)), (
            f"crew_memory value {memory_val!r} (type {type(memory_val).__name__}) "
            "is not serializable by OpenTelemetry"
        )
