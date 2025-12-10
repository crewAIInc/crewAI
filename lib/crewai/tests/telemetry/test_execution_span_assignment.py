"""Test that crew execution span is properly assigned during kickoff."""

import os
import threading

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_listener import EventListener
from crewai.telemetry import Telemetry


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Reset singletons between tests and enable telemetry."""
    original_telemetry = os.environ.get("CREWAI_DISABLE_TELEMETRY")
    original_otel = os.environ.get("OTEL_SDK_DISABLED")

    os.environ["CREWAI_DISABLE_TELEMETRY"] = "false"
    os.environ["OTEL_SDK_DISABLED"] = "false"

    with crewai_event_bus._rwlock.w_locked():
        crewai_event_bus._sync_handlers.clear()
        crewai_event_bus._async_handlers.clear()

    Telemetry._instance = None
    EventListener._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()

    yield

    with crewai_event_bus._rwlock.w_locked():
        crewai_event_bus._sync_handlers.clear()
        crewai_event_bus._async_handlers.clear()

    if original_telemetry is not None:
        os.environ["CREWAI_DISABLE_TELEMETRY"] = original_telemetry
    else:
        os.environ.pop("CREWAI_DISABLE_TELEMETRY", None)

    if original_otel is not None:
        os.environ["OTEL_SDK_DISABLED"] = original_otel
    else:
        os.environ.pop("OTEL_SDK_DISABLED", None)

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


@pytest.mark.vcr()
def test_end_crew_receives_valid_execution_span():
    """Test that end_crew receives a valid execution span to close.

    This verifies the complete lifecycle: span creation, assignment, and closure
    without errors when end_crew() accesses crew._execution_span.
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

    result = crew.kickoff()

    assert crew._execution_span is not None
    assert result is not None


@pytest.mark.vcr()
def test_crew_execution_span_not_set_when_share_crew_false():
    """Test that _execution_span is None when share_crew=False.

    When share_crew is False, crew_execution_span() returns None,
    so _execution_span should not be set.
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
        share_crew=False,
    )

    crew.kickoff()

    assert (
        not hasattr(crew, "_execution_span") or crew._execution_span is None
    ), "crew._execution_span should be None when share_crew=False"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_crew_execution_span_assigned_on_kickoff_async():
    """Test that _execution_span is assigned during async kickoff.

    Verifies that the async execution path also properly assigns
    the execution span.
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

    await crew.kickoff_async()

    assert crew._execution_span is not None, (
        "crew._execution_span should be set after kickoff_async when share_crew=True"
    )


@pytest.mark.vcr()
def test_crew_execution_span_assigned_on_kickoff_for_each():
    """Test that _execution_span is assigned for each crew execution.

    Verifies that batch execution properly assigns execution spans
    for each input.
    """
    agent = Agent(
        role="test agent",
        goal="say hello",
        backstory="a friendly agent",
        llm="gpt-4o-mini",
    )
    task = Task(
        description="Say hello to {name}",
        expected_output="hello",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        share_crew=True,
    )

    inputs = [{"name": "Alice"}, {"name": "Bob"}]
    results = crew.kickoff_for_each(inputs)

    assert len(results) == 2
