"""Test that crew execution span is properly assigned during kickoff."""

import threading

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_listener import EventListener
from crewai.telemetry import Telemetry


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons between tests."""
    Telemetry._instance = None
    EventListener._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()
    yield
    Telemetry._instance = None
    EventListener._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()


@pytest.mark.vcr()
def test_crew_execution_span_assigned_on_kickoff():
    """Test that _execution_span is assigned to crew after kickoff.

    The bug: event_listener.py calls crew_execution_span() but doesn't assign
    the returned span to source._execution_span, causing end_crew() to fail
    when it tries to access crew._execution_span.
    """
    agent = Agent(
        role="test agent",
        goal="say hello",
        backstory="a friendly agent",
        llm="gpt-4o-mini",
    )
    task = Task(
        description="Say hello",
        expected_output="hello",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        share_crew=True,
    )

    crew.kickoff()

    # The critical check: verify the crew has _execution_span set
    # This is what end_crew() needs to properly close the span
    assert crew._execution_span is not None, (
        "crew._execution_span should be set after kickoff when share_crew=True. "
        "The event_listener.py must assign the return value of crew_execution_span() "
        "to source._execution_span."
    )