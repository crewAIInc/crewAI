import os
from unittest.mock import MagicMock, patch

import pytest
from crewai.telemetry import Telemetry


@pytest.fixture(autouse=True)
def cleanup_telemetry():
    """Automatically clean up Telemetry singleton between tests."""
    Telemetry._instance = None
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
        env_var: value,
    }
    with patch.dict(os.environ, clean_env):
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
