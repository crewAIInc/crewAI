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


def test_signal_handler_registration_skipped_in_non_main_thread():
    """Test that signal handler registration is skipped when running from a non-main thread.

    This test verifies that when Telemetry is initialized from a non-main thread,
    the signal handler registration is skipped without raising noisy ValueError tracebacks.
    See: https://github.com/crewAIInc/crewAI/issues/4289
    """
    Telemetry._instance = None

    result = {"register_signal_handler_called": False, "error": None}

    def init_telemetry_in_thread():
        try:
            with patch("crewai.telemetry.telemetry.TracerProvider"):
                with patch.object(
                    Telemetry,
                    "_register_signal_handler",
                    wraps=lambda *args, **kwargs: None,
                ) as mock_register:
                    telemetry = Telemetry()
                    result["register_signal_handler_called"] = mock_register.called
                    result["telemetry"] = telemetry
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=init_telemetry_in_thread)
    thread.start()
    thread.join()

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert (
        result["register_signal_handler_called"] is False
    ), "Signal handler should not be registered in non-main thread"


def test_signal_handler_registration_skipped_logs_debug_message():
    """Test that a debug message is logged when signal handler registration is skipped.

    This test verifies that when Telemetry is initialized from a non-main thread,
    a debug message is logged indicating that signal handler registration was skipped.
    """
    Telemetry._instance = None

    result = {"telemetry": None, "error": None, "debug_calls": []}

    mock_logger_debug = MagicMock()

    def init_telemetry_in_thread():
        try:
            with patch("crewai.telemetry.telemetry.TracerProvider"):
                with patch(
                    "crewai.telemetry.telemetry.logger.debug", mock_logger_debug
                ):
                    result["telemetry"] = Telemetry()
                    result["debug_calls"] = [
                        str(call) for call in mock_logger_debug.call_args_list
                    ]
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=init_telemetry_in_thread)
    thread.start()
    thread.join()

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["telemetry"] is not None

    debug_calls = result["debug_calls"]
    assert any(
        "Skipping signal handler registration" in call for call in debug_calls
    ), f"Expected debug message about skipping signal handler registration, got: {debug_calls}"


def test_signal_handlers_registered_in_main_thread():
    """Test that signal handlers are registered when running from the main thread."""
    Telemetry._instance = None

    with patch("crewai.telemetry.telemetry.TracerProvider"):
        with patch(
            "crewai.telemetry.telemetry.Telemetry._register_signal_handler"
        ) as mock_register:
            telemetry = Telemetry()

            assert telemetry.ready is True
            assert mock_register.call_count >= 2
