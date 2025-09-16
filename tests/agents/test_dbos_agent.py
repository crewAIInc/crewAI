"""Test DBOSAgent creation and execution basic functionality."""

import os
from collections.abc import Generator, Iterator
from typing import Any

import pytest
from dbos import DBOS, DBOSConfig, SetWorkflowID

from crewai import Agent, Task
from crewai.durable_execution.dbos.dbos_agent import DBOSAgent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
from crewai.tools import tool

DBOS_SQLITE_FILE = "dbostest.sqlite"
DBOS_CONFIG: DBOSConfig = {
    "name": "pydantic_dbos_tests",
    "database_url": f"sqlite:///{DBOS_SQLITE_FILE}",
    "system_database_url": f"sqlite:///{DBOS_SQLITE_FILE}",
    "run_admin_server": False,
}


@pytest.fixture(scope="module")
def dbos() -> Generator[DBOS, Any, None]:
    dbos = DBOS(config=DBOS_CONFIG)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()


# Automatically clean up old DBOS sqlite files
@pytest.fixture(autouse=True, scope="module")
def cleanup_test_sqlite_file() -> Iterator[None]:
    if os.path.exists(DBOS_SQLITE_FILE):
        os.remove(DBOS_SQLITE_FILE)
    try:
        yield
    finally:
        if os.path.exists(DBOS_SQLITE_FILE):
            os.remove(DBOS_SQLITE_FILE)


def test_agent_creation(dbos: DBOS):
    agent = Agent(role="test role", goal="test goal", backstory="test backstory")

    assert agent.role == "test role"
    assert agent.goal == "test goal"
    assert agent.backstory == "test backstory"

    dbos_agent = DBOSAgent(
        orig_agent=agent,
        llm_step_config={"step_name": "llm_step"},
        function_calling_llm_step_config={"step_name": "func_call_step"},
    )

    assert dbos_agent.role == "test role"
    assert dbos_agent.goal == "test goal"
    assert dbos_agent.backstory == "test backstory"
    assert dbos_agent.orig_agent == agent
    assert (
        isinstance(dbos_agent._wrapped_agent, Agent)
        and dbos_agent._wrapped_agent != agent
    )
    assert dbos_agent.llm_step_config == {"step_name": "llm_step"}
    assert dbos_agent.function_calling_llm_step_config == {
        "step_name": "func_call_step"
    }


@pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_execution_with_tools(dbos: DBOS):
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
        orig_agent=orig_agent,
        llm_step_config={"step_name": "llm_step"},
        function_calling_llm_step_config={"step_name": "func_call_step"},
    )

    task = Task(
        description="What is 3 times 4?",
        agent=dbos_agent,
        expected_output="The result of the multiplication.",
    )
    received_events = []

    step_cnt = 0

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    @DBOS.step()
    def handle_tool_end(source, event):
        nonlocal step_cnt
        step_cnt += 1
        received_events.append(event)

    @DBOS.workflow(name="test_execution_with_tools")
    def run_task_with_dbos() -> str:
        return dbos_agent.execute_task(task)

    with SetWorkflowID("test_execution"):
        output = run_task_with_dbos()
    assert output == "The result of the multiplication is 12."

    assert len(received_events) == 1
    assert isinstance(received_events[0], ToolUsageFinishedEvent)
    assert received_events[0].tool_name == "multiplier"
    assert received_events[0].tool_args == {"first_number": 3, "second_number": 4}

    # Make sure DBOS correctly recorded the steps and workflows
    steps = DBOS.list_workflow_steps("test_execution")
    assert len(steps) == 2
    assert steps[0]["function_name"] == "dbos_agent_executor_invoke"
    assert steps[1]["function_name"] == "DBOS.getResult"
    assert step_cnt == 1

    # The child workflow is the agent's main execution loop
    child_steps = DBOS.list_workflow_steps(steps[0]["child_workflow_id"])
    assert len(child_steps) == 1
    assert "handle_tool_end" in child_steps[0]["function_name"]

    # Re-run the same workflow with the same ID
    with SetWorkflowID("test_execution"):
        output2 = run_task_with_dbos()
    assert output2 == "The result of the multiplication is 12."
    assert (
        step_cnt == 1
    )  # step count should not increase because DBOS has recorded the step
