"""Test that crew execution spans work correctly when crews run inside flows."""

import os
import threading

import pytest
from pydantic import BaseModel

from crewai import Agent, Crew, Task
from crewai.events.event_listener import EventListener
from crewai.flow.flow import Flow, listen, start
from crewai.telemetry import Telemetry


@pytest.fixture(autouse=True)
def enable_telemetry_for_tests():
    """Enable telemetry for these tests and reset singletons."""
    from crewai.events.event_bus import crewai_event_bus

    # Store original values
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

    Telemetry._instance = None
    EventListener._instance = None
    if hasattr(Telemetry, "_lock"):
        Telemetry._lock = threading.Lock()

    if original_telemetry is not None:
        os.environ["CREWAI_DISABLE_TELEMETRY"] = original_telemetry
    else:
        os.environ.pop("CREWAI_DISABLE_TELEMETRY", None)

    if original_otel is not None:
        os.environ["OTEL_SDK_DISABLED"] = original_otel
    else:
        os.environ.pop("OTEL_SDK_DISABLED", None)


class SimpleState(BaseModel):
    """Simple state for flow testing."""

    result: str = ""


@pytest.mark.vcr()
def test_crew_execution_span_in_flow_with_share_crew():
    """Test that crew._execution_span is properly set when crew runs inside a flow.

    This verifies that when a crew is kicked off inside a flow method with
    share_crew=True, the execution span is properly assigned and closed without
    errors.
    """

    class TestFlow(Flow[SimpleState]):
        @start()
        def run_crew(self):
            """Run a crew inside the flow."""
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

            assert crew._execution_span is not None, (
                "crew._execution_span should be set after kickoff even when "
                "crew runs inside a flow method"
            )

            self.state.result = str(result.raw)
            return self.state.result

    flow = TestFlow()
    flow.kickoff()

    assert flow.state.result != ""


@pytest.mark.vcr()
def test_crew_execution_span_not_set_in_flow_without_share_crew():
    """Test that crew._execution_span is None when share_crew=False in flow.

    Verifies that when a crew runs inside a flow with share_crew=False,
    no execution span is created.
    """

    class TestFlow(Flow[SimpleState]):
        @start()
        def run_crew(self):
            """Run a crew inside the flow without sharing."""
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

            result = crew.kickoff()

            assert (
                not hasattr(crew, "_execution_span") or crew._execution_span is None
            ), "crew._execution_span should be None when share_crew=False"

            self.state.result = str(result.raw)
            return self.state.result

    flow = TestFlow()
    flow.kickoff()

    assert flow.state.result != ""


@pytest.mark.vcr()
def test_multiple_crews_in_flow_span_lifecycle():
    """Test that multiple crews in a flow each get proper execution spans.

    This ensures that when multiple crews are executed sequentially in different
    flow methods, each crew gets its own execution span properly assigned and closed.
    """

    class MultiCrewFlow(Flow[SimpleState]):
        @start()
        def first_crew(self):
            """Run first crew."""
            agent = Agent(
                role="first agent",
                goal="first task",
                backstory="first agent",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="First task",
                expected_output="first result",
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                share_crew=True,
            )

            result = crew.kickoff()

            assert crew._execution_span is not None
            return str(result.raw)

        @listen(first_crew)
        def second_crew(self, first_result: str):
            """Run second crew."""
            agent = Agent(
                role="second agent",
                goal="second task",
                backstory="second agent",
                llm="gpt-4o-mini",
            )
            task = Task(
                description="Second task",
                expected_output="second result",
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                share_crew=True,
            )

            result = crew.kickoff()

            assert crew._execution_span is not None

            self.state.result = f"{first_result} + {result.raw}"
            return self.state.result

    flow = MultiCrewFlow()
    flow.kickoff()

    assert flow.state.result != ""


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_crew_execution_span_in_async_flow():
    """Test that crew execution spans work in async flow methods.

    Verifies that crews executed within async flow methods still properly
    assign and close execution spans.
    """

    class AsyncTestFlow(Flow[SimpleState]):
        @start()
        async def run_crew_async(self):
            """Run a crew inside an async flow method."""
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

            assert crew._execution_span is not None, (
                "crew._execution_span should be set in async flow method"
            )

            self.state.result = str(result.raw)
            return self.state.result

    flow = AsyncTestFlow()
    await flow.kickoff_async()

    assert flow.state.result != ""