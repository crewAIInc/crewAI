# conftest.py
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from dotenv import load_dotenv

load_result = load_dotenv(override=True)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with a temporary directory for SQLite storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the directory with proper permissions
        storage_dir = Path(temp_dir) / "crewai_test_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Validate that the directory was created successfully
        if not storage_dir.exists() or not storage_dir.is_dir():
            raise RuntimeError(
                f"Failed to create test storage directory: {storage_dir}"
            )

        # Verify directory permissions
        try:
            # Try to create a test file to verify write permissions
            test_file = storage_dir / ".permissions_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, IOError) as e:
            raise RuntimeError(
                f"Test storage directory {storage_dir} is not writable: {e}"
            )

        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        os.environ["CREWAI_TESTING"] = "true"
        yield

        os.environ.pop("CREWAI_TESTING", None)
        # Cleanup is handled automatically when tempfile context exits


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "telemetry: mark test as a telemetry test (don't mock telemetry)"
    )


@pytest.fixture(autouse=True)
def auto_mock_telemetry(request):
    if request.node.get_closest_marker("telemetry"):
        yield
        return

    if "telemetry" in str(request.fspath):
        yield
        return

    with patch.dict(
        os.environ, {"CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true"}
    ):
        with patch("crewai.telemetry.Telemetry") as mock_telemetry_class:
            mock_instance = create_mock_telemetry_instance()
            mock_telemetry_class.return_value = mock_instance

            with (
                patch(
                    "crewai.utilities.events.event_listener.Telemetry",
                    mock_telemetry_class,
                ),
                patch("crewai.tools.tool_usage.Telemetry", mock_telemetry_class),
                patch("crewai.cli.command.Telemetry", mock_telemetry_class),
                patch("crewai.cli.create_flow.Telemetry", mock_telemetry_class),
            ):
                yield mock_instance


def create_mock_telemetry_instance():
    mock_instance = Mock()

    mock_instance.ready = False
    mock_instance.trace_set = False
    mock_instance._initialized = True

    mock_instance._is_telemetry_disabled.return_value = True
    mock_instance._should_execute_telemetry.return_value = False

    telemetry_methods = [
        "set_tracer",
        "crew_creation",
        "task_started",
        "task_ended",
        "tool_usage",
        "tool_repeated_usage",
        "tool_usage_error",
        "crew_execution_span",
        "end_crew",
        "flow_creation_span",
        "flow_execution_span",
        "individual_test_result_span",
        "test_execution_span",
        "deploy_signup_error_span",
        "start_deployment_span",
        "create_crew_deployment_span",
        "get_crew_logs_span",
        "remove_crew_span",
        "flow_plotting_span",
        "_add_attribute",
        "_safe_telemetry_operation",
    ]

    for method in telemetry_methods:
        setattr(mock_instance, method, Mock(return_value=None))

    mock_instance.task_started.return_value = None

    return mock_instance


@pytest.fixture
def mock_opentelemetry_components():
    with (
        patch("opentelemetry.trace.get_tracer") as mock_get_tracer,
        patch("opentelemetry.trace.set_tracer_provider") as mock_set_provider,
        patch("opentelemetry.baggage.set_baggage") as mock_set_baggage,
        patch("opentelemetry.baggage.get_baggage") as mock_get_baggage,
        patch("opentelemetry.context.attach") as mock_attach,
        patch("opentelemetry.context.detach") as mock_detach,
    ):
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer

        yield {
            "get_tracer": mock_get_tracer,
            "set_tracer_provider": mock_set_provider,
            "tracer": mock_tracer,
            "span": mock_span,
            "set_baggage": mock_set_baggage,
            "get_baggage": mock_get_baggage,
            "attach": mock_attach,
            "detach": mock_detach,
        }


@pytest.fixture(scope="module")
def vcr_config(request) -> dict:
    return {
        "cassette_library_dir": "tests/cassettes",
        "record_mode": "new_episodes",
        "filter_headers": [("authorization", "AUTHORIZATION-XXX")],
    }
