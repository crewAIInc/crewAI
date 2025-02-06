import pytest
from datetime import datetime
from crewai.utilities.events.events import on, emit
from crewai.utilities.events.agent_events import (
    AgentExecutionStarted,
    AgentExecutionCompleted,
    AgentExecutionError,
)
from crewai.utilities.events.task_events import TaskStarted, TaskCompleted
from crewai.utilities.events.crew_events import CrewKickoffStarted, CrewKickoffCompleted
from crewai.crew import Crew
from crewai.agent import Agent
from crewai.task import Task
from unittest.mock import patch
from unittest import mock

base_agent = Agent(
    role="base_agent",
    llm="gpt-4o-mini",
    goal="Just say hi",
    backstory="You are a helpful assistant that just says hi",
)

base_task = Task(
    description="Just say hi",
    expected_output="hi",
    agent=base_agent,
)


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_start_kickoff_event():
    # Setup event listener
    received_events = []

    @on(CrewKickoffStarted)
    def handle_crew_start(source, event):
        received_events.append(event)

    # Create a simple crew
    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    # Run the crew
    crew.kickoff()

    # Verify the event was emitted
    assert len(received_events) == 1
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_started"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_end_kickoff_event():
    # Setup event listener
    received_events = []

    @on(CrewKickoffCompleted)
    def handle_crew_end(source, event):
        received_events.append(event)

    # Create a simple crew
    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    # Run the crew
    crew.kickoff()

    # Verify the event was emitted
    assert len(received_events) == 1
    assert received_events[0].crew_name == "TestCrew"
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "crew_kickoff_completed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_start_task_event():
    # Setup event listener
    received_events = []

    @on(TaskStarted)
    def handle_task_start(source, event):
        received_events.append(event)

    # Create a simple crew
    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    # Run the crew
    crew.kickoff()

    # Verify the event was emitted
    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "task_started"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_emits_end_task_event():
    # Setup event listener
    received_events = []

    @on(TaskCompleted)
    def handle_task_end(source, event):
        received_events.append(event)

    # Create a simple crew
    crew = Crew(agents=[base_agent], tasks=[base_task], name="TestCrew")

    # Run the crew
    crew.kickoff()

    # Verify the event was emitted
    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "task_completed"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_emits_execution_error_event():
    # Setup event listener
    received_events = []

    @on(AgentExecutionError)
    def handle_agent_error(source, event):
        received_events.append(event)

    # Create an agent that will fail
    failing_agent = Agent(
        role="failing_agent",
        goal="Fail execution",
        backstory="You are an agent that will fail",
        max_retry_limit=1,  # Set low retry limit for testing
    )

    # Create a task that will trigger an error
    failing_task = Task(
        description="This will fail", agent=failing_agent, expected_output="hi"
    )

    error_message = "Forced error for testing"
    # Mock the agent executor to raise an exception
    with patch.object(failing_agent.agent_executor, "invoke") as mock_invoke:
        mock_invoke.side_effect = Exception(error_message)
        assert failing_agent._times_executed == 0
        assert failing_agent.max_retry_limit == 1

        # Execute task which should fail and emit error
        with pytest.raises(Exception) as e:
            failing_agent.execute_task(failing_task)

        print("error message: ", e.value.args[0])

        # assert e.value.args[0] == error_message
        # assert failing_agent._times_executed == 2  # Initial attempt + 1 retry

        # Verify the invoke was called twice (initial + retry)
        mock_invoke.assert_has_calls(
            [
                mock.call(
                    {
                        "input": "This will fail\n\nThis is the expect criteria for your final answer: hi\nyou MUST return the actual complete content as the final answer, not a summary.",
                        "tool_names": "",
                        "tools": "",
                        "ask_for_human_input": False,
                    }
                ),
                mock.call(
                    {
                        "input": "This will fail\n\nThis is the expect criteria for your final answer: hi\nyou MUST return the actual complete content as the final answer, not a summary.",
                        "tool_names": "",
                        "tools": "",
                        "ask_for_human_input": False,
                    }
                ),
            ]
        )
    print("made it here")

    # Verify the error event was emitted
    assert len(received_events) == 1
    assert isinstance(received_events[0].timestamp, datetime)
    assert received_events[0].type == "agent_execution_error"
    assert received_events[0].agent == failing_agent
    assert received_events[0].task == failing_task
    assert error_message in received_events[0].error
