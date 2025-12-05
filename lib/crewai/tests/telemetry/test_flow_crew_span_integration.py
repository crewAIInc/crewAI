"""Test that crew execution spans work correctly when crews run inside flows.

Note: These tests use mocked LLM responses instead of VCR cassettes because
VCR's httpx async stubs have a known incompatibility with the OpenAI client
when running inside asyncio.run() (which Flow.kickoff() uses). The VCR
assertion `assert not hasattr(resp, "_decoder")` fails silently when the
OpenAI client reads responses before VCR can serialize them.
"""

import os
import threading
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from crewai import Agent, Crew, Task, LLM
from crewai.events.event_listener import EventListener
from crewai.flow.flow import Flow, listen, start
from crewai.telemetry import Telemetry
from crewai.types.usage_metrics import UsageMetrics


class SimpleState(BaseModel):
    """Simple state for flow testing."""

    result: str = ""


def create_mock_llm() -> Mock:
    """Create a mock LLM that returns a simple response.

    The mock includes all attributes required by the telemetry system,
    particularly the 'model' attribute which is accessed during span creation.
    """
    mock_llm = Mock(spec=LLM)
    mock_llm.call.return_value = "Hello! This is a test response."
    mock_llm.stop = []
    mock_llm.model = "gpt-4o-mini"  # Required by telemetry
    mock_llm.supports_stop_words.return_value = True
    mock_llm.get_token_usage_summary.return_value = UsageMetrics(
        total_tokens=100,
        prompt_tokens=50,
        completion_tokens=50,
        cached_prompt_tokens=0,
        successful_requests=1,
    )
    return mock_llm


@pytest.fixture(autouse=True)
def enable_telemetry_for_tests():
    """Enable telemetry for these tests and reset singletons."""
    from crewai.events.event_bus import crewai_event_bus

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


def test_crew_execution_span_in_flow_with_share_crew():
    """Test that crew._execution_span is properly set when crew runs inside a flow.

    This verifies that when a crew is kicked off inside a flow method with
    share_crew=True, the execution span is properly assigned and closed without
    errors.
    """
    mock_llm = create_mock_llm()

    class SampleFlow(Flow[SimpleState]):
        @start()
        def run_crew(self):
            """Run a crew inside the flow."""
            agent = Agent(
                role="test agent",
                goal="say hello",
                backstory="a friendly agent",
                llm=mock_llm,
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

    flow = SampleFlow()
    flow.kickoff()

    assert flow.state.result != ""
    mock_llm.call.assert_called()


def test_crew_execution_span_not_set_in_flow_without_share_crew():
    """Test that crew._execution_span is None when share_crew=False in flow.

    Verifies that when a crew runs inside a flow with share_crew=False,
    no execution span is created.
    """
    mock_llm = create_mock_llm()

    class SampleTestFlowNotSet(Flow[SimpleState]):
        @start()
        def run_crew(self):
            """Run a crew inside the flow without sharing."""
            agent = Agent(
                role="test agent",
                goal="say hello",
                backstory="a friendly agent",
                llm=mock_llm,
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

    flow = SampleTestFlowNotSet()
    flow.kickoff()

    assert flow.state.result != ""
    mock_llm.call.assert_called()


def test_multiple_crews_in_flow_span_lifecycle():
    """Test that multiple crews in a flow each get proper execution spans.

    This ensures that when multiple crews are executed sequentially in different
    flow methods, each crew gets its own execution span properly assigned and closed.
    """
    mock_llm_1 = create_mock_llm()
    mock_llm_1.call.return_value = "First crew result"

    mock_llm_2 = create_mock_llm()
    mock_llm_2.call.return_value = "Second crew result"

    class SampleMultiCrewFlow(Flow[SimpleState]):
        @start()
        def first_crew(self):
            """Run first crew."""
            agent = Agent(
                role="first agent",
                goal="first task",
                backstory="first agent",
                llm=mock_llm_1,
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
                llm=mock_llm_2,
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

    flow = SampleMultiCrewFlow()
    flow.kickoff()

    assert flow.state.result != ""
    assert "+" in flow.state.result
    mock_llm_1.call.assert_called()
    mock_llm_2.call.assert_called()


@pytest.mark.asyncio
async def test_crew_execution_span_in_async_flow():
    """Test that crew execution spans work in async flow methods.

    Verifies that crews executed within async flow methods still properly
    assign and close execution spans.
    """
    mock_llm = create_mock_llm()

    class AsyncTestFlow(Flow[SimpleState]):
        @start()
        async def run_crew_async(self):
            """Run a crew inside an async flow method."""
            agent = Agent(
                role="test agent",
                goal="say hello",
                backstory="a friendly agent",
                llm=mock_llm,
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
    mock_llm.call.assert_called()