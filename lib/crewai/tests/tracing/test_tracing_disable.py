"""Tests for CREWAI_DISABLE_TELEMETRY affecting tracing system."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from crewai.events.listeners.tracing.trace_batch_manager import TraceBatchManager
from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
from crewai.events.listeners.tracing.utils import is_tracking_disabled


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset TraceCollectionListener singleton between tests."""
    TraceCollectionListener._instance = None
    TraceCollectionListener._initialized = False
    TraceCollectionListener._listeners_setup = False
    yield
    TraceCollectionListener._instance = None
    TraceCollectionListener._initialized = False
    TraceCollectionListener._listeners_setup = False


@pytest.fixture
def mock_plus_api():
    """Mock PlusAPI to prevent actual network calls."""
    with patch("crewai.events.listeners.tracing.trace_batch_manager.PlusAPI") as mock:
        api_instance = MagicMock()
        mock.return_value = api_instance
        yield api_instance


@pytest.mark.parametrize(
    "env_var,value,expected_disabled",
    [
        ("CREWAI_DISABLE_TELEMETRY", "true", True),
        ("CREWAI_DISABLE_TELEMETRY", "TRUE", True),
        ("CREWAI_DISABLE_TELEMETRY", "True", True),
        ("CREWAI_DISABLE_TRACKING", "true", True),
        ("CREWAI_DISABLE_TRACKING", "TRUE", True),
        ("CREWAI_DISABLE_TELEMETRY", "false", False),
        ("CREWAI_DISABLE_TRACKING", "false", False),
    ],
)
def test_is_tracking_disabled_env_vars(env_var, value, expected_disabled):
    """Test is_tracking_disabled() with different environment variables."""
    with patch.dict(os.environ, {env_var: value}, clear=True):
        assert is_tracking_disabled() == expected_disabled


def test_is_tracking_disabled_default():
    """Test is_tracking_disabled() returns False by default."""
    with patch.dict(os.environ, {}, clear=True):
        assert is_tracking_disabled() is False


def test_trace_batch_manager_initialize_backend_batch_disabled(mock_plus_api):
    """Test that _initialize_backend_batch does not make network calls when disabled."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            manager = TraceBatchManager()
            manager.current_batch = MagicMock()
            manager.current_batch.batch_id = "test_batch_id"

            manager._initialize_backend_batch(
                user_context={"user_id": "test"},
                execution_metadata={"execution_type": "crew"},
                use_ephemeral=False,
            )

            mock_plus_api.initialize_trace_batch.assert_not_called()
            mock_plus_api.initialize_ephemeral_trace_batch.assert_not_called()


def test_trace_batch_manager_initialize_backend_batch_ephemeral_disabled(
    mock_plus_api,
):
    """Test that ephemeral batch initialization does not make network calls when disabled."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            manager = TraceBatchManager()
            manager.current_batch = MagicMock()
            manager.current_batch.batch_id = "test_batch_id"

            manager._initialize_backend_batch(
                user_context={"user_id": "test"},
                execution_metadata={"execution_type": "crew"},
                use_ephemeral=True,
            )

            mock_plus_api.initialize_trace_batch.assert_not_called()
            mock_plus_api.initialize_ephemeral_trace_batch.assert_not_called()


def test_trace_batch_manager_send_events_disabled(mock_plus_api):
    """Test that _send_events_to_backend returns success without making calls when disabled."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            manager = TraceBatchManager()
            manager.trace_batch_id = "test_batch_id"
            manager.event_buffer = [MagicMock()]

            result = manager._send_events_to_backend()

            assert result == 200
            mock_plus_api.send_trace_events.assert_not_called()
            mock_plus_api.send_ephemeral_trace_events.assert_not_called()


def test_trace_batch_manager_finalize_batch_disabled(mock_plus_api):
    """Test that finalize_batch returns None without making calls when disabled."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            manager = TraceBatchManager()
            manager.current_batch = MagicMock()

            result = manager.finalize_batch()

            assert result is None
            mock_plus_api.finalize_trace_batch.assert_not_called()
            mock_plus_api.finalize_ephemeral_trace_batch.assert_not_called()
            mock_plus_api.mark_trace_batch_as_failed.assert_not_called()


def test_trace_batch_manager_finalize_backend_batch_disabled(mock_plus_api):
    """Test that _finalize_backend_batch does not make network calls when disabled."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            manager = TraceBatchManager()
            manager.trace_batch_id = "test_batch_id"

            manager._finalize_backend_batch(events_count=5)

            mock_plus_api.finalize_trace_batch.assert_not_called()
            mock_plus_api.finalize_ephemeral_trace_batch.assert_not_called()


def test_trace_collection_listener_init_disabled():
    """Test that TraceCollectionListener initialization is skipped when disabled."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        listener = TraceCollectionListener()

        assert listener._initialized is True
        assert not hasattr(listener, "batch_manager")
        assert not hasattr(listener, "first_time_handler")


def test_trace_collection_listener_setup_listeners_disabled():
    """Test that setup_listeners does not register handlers when disabled."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        listener = TraceCollectionListener()
        mock_event_bus = MagicMock()

        listener.setup_listeners(mock_event_bus)

        assert listener._listeners_setup is True
        mock_event_bus.on.assert_not_called()


def test_trace_batch_manager_enabled_makes_calls(mock_plus_api):
    """Test that network calls ARE made when tracking is enabled (negative test)."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "false"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            with patch(
                "crewai.events.listeners.tracing.trace_batch_manager.should_auto_collect_first_time_traces"
            ) as mock_first_time:
                mock_first_time.return_value = False

                mock_response = Mock()
                mock_response.status_code = 201
                mock_response.json.return_value = {"trace_id": "test_trace_id"}
                mock_plus_api.initialize_trace_batch.return_value = mock_response

                manager = TraceBatchManager()
                manager.current_batch = MagicMock()
                manager.current_batch.batch_id = "test_batch_id"
                manager.current_batch.version = "1.0.0"

                manager._initialize_backend_batch(
                    user_context={"user_id": "test"},
                    execution_metadata={"execution_type": "crew"},
                    use_ephemeral=False,
                )

                mock_plus_api.initialize_trace_batch.assert_called_once()


def test_trace_batch_manager_enabled_ephemeral_makes_calls(mock_plus_api):
    """Test that ephemeral network calls ARE made when tracking is enabled (negative test)."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "false"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            with patch(
                "crewai.events.listeners.tracing.trace_batch_manager.should_auto_collect_first_time_traces"
            ) as mock_first_time:
                mock_first_time.return_value = False

                mock_response = Mock()
                mock_response.status_code = 201
                mock_response.json.return_value = {
                    "ephemeral_trace_id": "test_ephemeral_id"
                }
                mock_plus_api.initialize_ephemeral_trace_batch.return_value = (
                    mock_response
                )

                manager = TraceBatchManager()
                manager.current_batch = MagicMock()
                manager.current_batch.batch_id = "test_batch_id"
                manager.current_batch.version = "1.0.0"

                manager._initialize_backend_batch(
                    user_context={"user_id": "test"},
                    execution_metadata={"execution_type": "crew"},
                    use_ephemeral=True,
                )

                mock_plus_api.initialize_ephemeral_trace_batch.assert_called_once()


def test_trace_collection_listener_enabled_registers_handlers():
    """Test that handlers ARE registered when tracking is enabled (negative test)."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "false"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ):
            with patch(
                "crewai.events.listeners.tracing.trace_batch_manager.PlusAPI"
            ):
                listener = TraceCollectionListener()
                
                assert hasattr(listener, "batch_manager")
                assert hasattr(listener, "first_time_handler")
                
                listener._listeners_setup = False
                
                with patch.object(
                    listener, "_register_flow_event_handlers"
                ) as mock_flow:
                    with patch.object(
                        listener, "_register_context_event_handlers"
                    ) as mock_context:
                        with patch.object(
                            listener, "_register_action_event_handlers"
                        ) as mock_action:
                            mock_event_bus = MagicMock()
                            listener.setup_listeners(mock_event_bus)

                            mock_flow.assert_called_once_with(mock_event_bus)
                            mock_context.assert_called_once_with(mock_event_bus)
                            mock_action.assert_called_once_with(mock_event_bus)
                            assert listener._listeners_setup is True


def test_crewai_disable_tracking_also_works():
    """Test that CREWAI_DISABLE_TRACKING also disables tracing."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TRACKING": "true"}):
        with patch(
            "crewai.events.listeners.tracing.trace_batch_manager.get_auth_token"
        ) as mock_auth:
            mock_auth.return_value = "test_token"
            with patch(
                "crewai.events.listeners.tracing.trace_batch_manager.PlusAPI"
            ) as mock_plus_api:
                api_instance = MagicMock()
                mock_plus_api.return_value = api_instance

                manager = TraceBatchManager()
                manager.current_batch = MagicMock()
                manager.current_batch.batch_id = "test_batch_id"

                manager._initialize_backend_batch(
                    user_context={"user_id": "test"},
                    execution_metadata={"execution_type": "crew"},
                    use_ephemeral=False,
                )

                api_instance.initialize_trace_batch.assert_not_called()
                api_instance.initialize_ephemeral_trace_batch.assert_not_called()
