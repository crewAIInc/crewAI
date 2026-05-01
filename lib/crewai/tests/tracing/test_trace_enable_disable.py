"""Tests to verify that traces are sent when enabled and not sent when disabled.

VCR will record HTTP interactions. Inspect cassettes to verify tracing behavior.
"""

from unittest.mock import patch

import pytest
from crewai import Agent, Crew, Task
from crewai.events.listeners.tracing.utils import (
    should_suppress_tracing_messages,
)
from tests.utils import wait_for_event_handlers


def test_should_suppress_tracing_messages_via_env(monkeypatch):
    monkeypatch.setenv("CREWAI_SUPPRESS_TRACING_MESSAGES", "true")

    assert should_suppress_tracing_messages() is True


def test_should_suppress_tracing_messages_when_user_declined(monkeypatch):
    monkeypatch.delenv("CREWAI_SUPPRESS_TRACING_MESSAGES", raising=False)

    with patch(
        "crewai.events.listeners.tracing.utils._load_user_data",
        return_value={"first_execution_done": True, "trace_consent": False},
    ):
        assert should_suppress_tracing_messages() is True


class TestTraceEnableDisable:
    """Test suite to verify trace sending behavior with VCR cassette recording."""

    @pytest.mark.vcr()
    def test_no_http_calls_when_disabled_via_env(self):
        """Test execution when tracing disabled via CREWAI_TRACING_ENABLED=false."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CREWAI_TRACING_ENABLED", "false")
            mp.setenv("CREWAI_DISABLE_TELEMETRY", "false")

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello",
                expected_output="hello",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)

            result = crew.kickoff()
            wait_for_event_handlers()

            assert result is not None

    @pytest.mark.vcr()
    def test_no_http_calls_when_disabled_via_tracing_false(self):
        """Test execution when tracing=False explicitly set."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CREWAI_DISABLE_TELEMETRY", "false")

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello",
                expected_output="hello",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False, tracing=False)

            result = crew.kickoff()
            wait_for_event_handlers()

            assert result is not None

    @pytest.mark.vcr()
    def test_trace_calls_when_enabled_via_env(self):
        """Test execution when tracing enabled via CREWAI_TRACING_ENABLED=true."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CREWAI_TRACING_ENABLED", "true")
            mp.setenv("CREWAI_DISABLE_TELEMETRY", "false")
            mp.setenv("OTEL_SDK_DISABLED", "false")

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello",
                expected_output="hello",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False)

            result = crew.kickoff()
            wait_for_event_handlers()

            assert result is not None

    @pytest.mark.vcr()
    def test_trace_calls_when_enabled_via_tracing_true(self):
        """Test execution when tracing=True explicitly set."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CREWAI_DISABLE_TELEMETRY", "false")
            mp.setenv("OTEL_SDK_DISABLED", "false")

            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Say hello",
                expected_output="hello",
                agent=agent,
            )
            crew = Crew(agents=[agent], tasks=[task], verbose=False, tracing=True)

            result = crew.kickoff()
            wait_for_event_handlers()

            assert result is not None
