import os
import pytest
from unittest.mock import patch

from crewai.telemetry import Telemetry


def test_telemetry_disabled_with_otel_sdk_disabled():
    """Test that telemetry is disabled when OTEL_SDK_DISABLED is set to true."""
    with patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
        telemetry = Telemetry()
        assert telemetry.ready is False


def test_telemetry_disabled_with_crewai_disable_telemetry():
    """Test that telemetry is disabled when CREWAI_DISABLE_TELEMETRY is set to true."""
    with patch.dict(os.environ, {"CREWAI_DISABLE_TELEMETRY": "true"}):
        telemetry = Telemetry()
        assert telemetry.ready is False


def test_telemetry_enabled_by_default():
    """Test that telemetry is enabled by default."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("crewai.telemetry.telemetry.TracerProvider"):
            telemetry = Telemetry()
            assert telemetry.ready is True
