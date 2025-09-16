"""Test DBOSAgent creation and execution basic functionality."""

from crewai import Agent
from crewai.durable_execution.dbos.agent import DBOSAgent


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
