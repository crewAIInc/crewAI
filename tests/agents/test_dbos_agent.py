"""Test DBOSAgent creation and execution basic functionality."""

import pytest

from crewai import Agent, Task
from crewai.durable_execution.dbos.agent import DBOSAgent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.tools import tool


def test_agent_creation():
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"

    dbos_agent = DBOSAgent(
        wrapped_agent=agent,
        llm_step_config={"step_name": "llm_step"},
        function_calling_llm_step_config={"step_name": "func_call_step"},
    )

    assert dbos_agent.role == "test role"
    assert dbos_agent.goal == "test goal"
    assert dbos_agent.backstory == "test backstory"
    assert dbos_agent.llm == agent.llm
    assert dbos_agent.function_calling_llm == agent.function_calling_llm
    assert dbos_agent.wrapped_agent == agent
    assert dbos_agent.llm_step_config == {"step_name": "llm_step"}
    assert dbos_agent.function_calling_llm_step_config == {
        "step_name": "func_call_step"
    }


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_tools():
    @tool
    def multiplier(first_number: int, second_number: int) -> float:
        """Useful for when you need to multiply two numbers together."""
        return first_number * second_number

    orig_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[multiplier],
        allow_delegation=False,
    )
    dbos_agent = DBOSAgent(
        wrapped_agent=orig_agent,
        llm_step_config={"step_name": "llm_step"},
        function_calling_llm_step_config={"step_name": "func_call_step"},
    )

    task = Task(
        description="What is 3 times 4?",
        agent=dbos_agent,
        expected_output="The result of the multiplication.",
    )
    received_events = []

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def handle_tool_end(source, event):
        received_events.append(event)

    output = dbos_agent.execute_task(task)
    assert output == "The result of the multiplication is 12."

    assert len(received_events) == 1
    assert isinstance(received_events[0], ToolUsageFinishedEvent)
    assert received_events[0].tool_name == "multiplier"
    assert received_events[0].tool_args == {"first_number": 3, "second_number": 4}
