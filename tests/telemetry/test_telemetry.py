import os
from unittest.mock import patch

import pytest

from crewai import Agent, Crew, Task
from crewai.telemetry import Telemetry
from opentelemetry import trace


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
    with patch.dict(os.environ, {env_var: value}):
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
@pytest.mark.vcr(filter_headers=["authorization"])
def test_telemetry_fails_due_connect_timeout(export_mock, logger_mock):
    error = Exception("Test exception")
    export_mock.side_effect = error

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

    export_mock.assert_called_once()
    logger_mock.assert_called_once_with(error)
