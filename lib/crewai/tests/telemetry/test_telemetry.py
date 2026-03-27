import os
import threading
from unittest.mock import Mock, patch

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


@pytest.mark.telemetry
class TestCrewMemoryTelemetrySerialization:
    """Regression tests for https://github.com/crewAIInc/crewAI/issues/4703"""

    def _create_telemetry(self):
        with patch.dict(
            os.environ,
            {"CREWAI_DISABLE_TELEMETRY": "false", "OTEL_SDK_DISABLED": "false"},
        ):
            with patch("crewai.telemetry.telemetry.TracerProvider"):
                return Telemetry()

    def _make_mock_crew(self, memory_value):
        crew = Mock()
        crew.memory = memory_value
        crew.share_crew = False
        crew.agents = []
        crew.tasks = []
        crew.process = "sequential"
        crew.fingerprint = None
        return crew

    def test_custom_memory_object_does_not_crash(self):
        """crew_creation should not raise when crew.memory is a non-bool object."""
        telemetry = self._create_telemetry()
        crew = self._make_mock_crew(memory_value=Mock())

        # Should not raise; before the fix this would cause:
        # "Invalid type Mock for attribute 'crew_memory' value."
        telemetry.crew_creation(crew, inputs=None)

    def test_custom_memory_object_serialized_as_bool(self):
        """crew.memory should be converted to a bool for the telemetry span."""
        telemetry = self._create_telemetry()
        memory_object = Mock()  # truthy non-bool
        crew = self._make_mock_crew(memory_value=memory_object)

        captured = {}
        original_add = telemetry._add_attribute

        def spy_add(span, key, value):
            captured[key] = value
            original_add(span, key, value)

        telemetry._add_attribute = spy_add
        telemetry.crew_creation(crew, inputs=None)

        assert "crew_memory" in captured
        assert captured["crew_memory"] is True
        assert type(captured["crew_memory"]) is bool

    def test_bool_memory_passes_through_unchanged(self):
        """crew.memory=True should remain True (not get re-wrapped)."""
        telemetry = self._create_telemetry()
        crew = self._make_mock_crew(memory_value=True)

        captured = {}
        original_add = telemetry._add_attribute

        def spy_add(span, key, value):
            captured[key] = value
            original_add(span, key, value)

        telemetry._add_attribute = spy_add
        telemetry.crew_creation(crew, inputs=None)

        assert captured["crew_memory"] is True

    def test_false_memory_passes_through_unchanged(self):
        """crew.memory=False should remain False."""
        telemetry = self._create_telemetry()
        crew = self._make_mock_crew(memory_value=False)

        captured = {}
        original_add = telemetry._add_attribute

        def spy_add(span, key, value):
            captured[key] = value
            original_add(span, key, value)

        telemetry._add_attribute = spy_add
        telemetry.crew_creation(crew, inputs=None)

        assert captured["crew_memory"] is False
