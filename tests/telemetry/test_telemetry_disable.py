import os
from unittest.mock import patch

import pytest

from crewai.telemetry import Telemetry


@pytest.mark.parametrize(("env_var", "value", "expected_ready"), [
    ("OTEL_SDK_DISABLED", "true", False),
    ("OTEL_SDK_DISABLED", "TRUE", False),
    ("CREWAI_DISABLE_TELEMETRY", "true", False),
    ("CREWAI_DISABLE_TELEMETRY", "TRUE", False),
    ("OTEL_SDK_DISABLED", "false", True),
    ("CREWAI_DISABLE_TELEMETRY", "false", True),
])
def test_telemetry_environment_variables(env_var, value, expected_ready) -> None:
    """Test telemetry state with different environment variable configurations."""
    with patch.dict(os.environ, {env_var: value}):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is expected_ready


def test_telemetry_enabled_by_default() -> None:
    """Test that telemetry is enabled by default."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is True
