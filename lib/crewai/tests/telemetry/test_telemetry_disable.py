import os
from unittest.mock import MagicMock, patch

import pytest
from crewai.events.listeners.tracing.utils import _tracing_enabled
from crewai.telemetry import Telemetry


@pytest.fixture(autouse=True)
def cleanup_telemetry():
    """Automatically clean up Telemetry singleton between tests."""
    Telemetry._instance = None
    with patch(
        "crewai.telemetry.telemetry.has_user_declined_tracing",
        return_value=False,
    ):
        yield
    Telemetry._instance = None


@pytest.mark.telemetry
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
    # Clear all telemetry-related env vars first, then set the one under test
    clean_env = {
        "OTEL_SDK_DISABLED": "false",
        "CREWAI_DISABLE_TELEMETRY": "false",
        "CREWAI_DISABLE_TRACKING": "false",
        "CREWAI_TRACING_ENABLED": "",
        env_var: value,
    }
    with patch.dict(os.environ, clean_env, clear=True):
        telemetry = Telemetry()
        assert telemetry.ready is expected_ready


@pytest.mark.telemetry
def test_telemetry_enabled_by_default():
    """Test that telemetry is enabled by default."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is True


@pytest.mark.telemetry
def test_telemetry_disable_after_singleton_creation():
    """Test that telemetry operations are disabled when env var is set after singleton creation."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is True

            mock_operation = MagicMock()
            telemetry._safe_telemetry_operation(mock_operation)
            mock_operation.assert_called_once()

            mock_operation.reset_mock()

            os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

            telemetry._safe_telemetry_operation(mock_operation)
            mock_operation.assert_not_called()


@pytest.mark.telemetry
def test_telemetry_disable_with_multiple_instances():
    """Test that multiple telemetry instances respect dynamically changed env vars."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry1 = Telemetry()
            assert telemetry1.ready is True

            os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

            telemetry2 = Telemetry()
            assert telemetry2 is telemetry1
            assert telemetry2.ready is True

            mock_operation = MagicMock()
            telemetry2._safe_telemetry_operation(mock_operation)
            mock_operation.assert_not_called()


@pytest.mark.telemetry
def test_telemetry_otel_sdk_disabled_after_creation():
    """Test that OTEL_SDK_DISABLED also works when set after singleton creation."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is True

            mock_operation = MagicMock()
            telemetry._safe_telemetry_operation(mock_operation)
            mock_operation.assert_called_once()

            mock_operation.reset_mock()

            os.environ["OTEL_SDK_DISABLED"] = "true"

            telemetry._safe_telemetry_operation(mock_operation)
            mock_operation.assert_not_called()


@pytest.mark.telemetry
@pytest.mark.parametrize(
    "tracing_value",
    ["false", "False", "FALSE", "0"],
)
def test_telemetry_disabled_when_crewai_tracing_enabled_is_false(tracing_value):
    """Test that telemetry is disabled when CREWAI_TRACING_ENABLED is explicitly false.

    Regression test for https://github.com/crewAIInc/crewAI/issues/4525
    """
    clean_env = {
        "OTEL_SDK_DISABLED": "false",
        "CREWAI_DISABLE_TELEMETRY": "false",
        "CREWAI_DISABLE_TRACKING": "false",
        "CREWAI_TRACING_ENABLED": tracing_value,
    }
    with patch.dict(os.environ, clean_env):
        telemetry = Telemetry()
        assert telemetry.ready is False


@pytest.mark.telemetry
def test_telemetry_not_disabled_when_crewai_tracing_enabled_unset():
    """Test that telemetry remains enabled when CREWAI_TRACING_ENABLED is not set."""
    clean_env = {
        "OTEL_SDK_DISABLED": "false",
        "CREWAI_DISABLE_TELEMETRY": "false",
        "CREWAI_DISABLE_TRACKING": "false",
    }
    with patch.dict(os.environ, clean_env, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            with patch(
                "crewai.telemetry.telemetry.has_user_declined_tracing",
                return_value=False,
            ):
                telemetry = Telemetry()
                assert telemetry.ready is True


@pytest.mark.telemetry
def test_telemetry_disabled_when_user_declined_tracing():
    """Test that telemetry is disabled when user has declined tracing via first-time prompt.

    Regression test for https://github.com/crewAIInc/crewAI/issues/4525
    """
    clean_env = {
        "OTEL_SDK_DISABLED": "false",
        "CREWAI_DISABLE_TELEMETRY": "false",
        "CREWAI_DISABLE_TRACKING": "false",
    }
    with patch.dict(os.environ, clean_env, clear=True):
        with patch(
            "crewai.telemetry.telemetry.has_user_declined_tracing",
            return_value=True,
        ) as mock_declined:
            telemetry = Telemetry()
            mock_declined.assert_called()
            assert telemetry.ready is False


@pytest.mark.telemetry
def test_telemetry_operations_blocked_when_crewai_tracing_enabled_false_after_init():
    """Test that telemetry operations are blocked when CREWAI_TRACING_ENABLED=false set after init.

    Regression test for https://github.com/crewAIInc/crewAI/issues/4525
    """
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is True

            mock_operation = MagicMock()
            telemetry._safe_telemetry_operation(mock_operation)
            mock_operation.assert_called_once()

            mock_operation.reset_mock()

            os.environ["CREWAI_TRACING_ENABLED"] = "false"

            telemetry._safe_telemetry_operation(mock_operation)
            mock_operation.assert_not_called()


@pytest.mark.telemetry
def test_telemetry_operations_allowed_when_tracing_context_true():
    """Test that telemetry operations are allowed when tracing context var is True."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            with patch(
                "crewai.telemetry.telemetry.has_user_declined_tracing",
                return_value=False,
            ):
                telemetry = Telemetry()
                assert telemetry.ready is True

                mock_operation = MagicMock()

                token = _tracing_enabled.set(True)
                try:
                    telemetry._safe_telemetry_operation(mock_operation)
                    mock_operation.assert_called_once()
                finally:
                    _tracing_enabled.reset(token)


@pytest.mark.telemetry
def test_telemetry_operations_allowed_when_tracing_context_none():
    """Test that telemetry operations are allowed when tracing context var is None (default)."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            with patch(
                "crewai.telemetry.telemetry.has_user_declined_tracing",
                return_value=False,
            ):
                telemetry = Telemetry()
                assert telemetry.ready is True

                mock_operation = MagicMock()
                telemetry._safe_telemetry_operation(mock_operation)
                mock_operation.assert_called_once()
