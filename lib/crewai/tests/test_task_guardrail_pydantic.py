"""A task with output_pydantic and a guardrail must still expose .pydantic
inside the guardrail (#9495)."""

from unittest.mock import Mock

from pydantic import BaseModel

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput


class Report(BaseModel):
    score: int


def _mock_agent() -> Mock:
    agent = Mock()
    agent.role = "test_agent"
    agent.crew = None
    agent.last_messages = []
    return agent


def test_pydantic_output_is_set_inside_a_guardrail():
    """The guardrail sees the exported structured output of the first attempt."""
    agent = _mock_agent()
    agent.execute_task.return_value = '{"score": 7}'
    seen: dict = {}

    def guardrail(result: TaskOutput):
        seen["pydantic"] = result.pydantic
        return (True, result.raw)

    task = Task(
        description="make a report",
        expected_output="a report",
        output_pydantic=Report,
        guardrail=guardrail,
    )

    result = task.execute_sync(agent=agent)

    assert isinstance(seen["pydantic"], Report), seen
    assert seen["pydantic"].score == 7
    assert result.pydantic is not None


def test_guardrails_list_also_sees_the_pydantic_output():
    agent = _mock_agent()
    agent.execute_task.return_value = '{"score": 9}'
    seen: list = []

    def guardrail(result: TaskOutput):
        seen.append(result.pydantic)
        return (True, result.raw)

    task = Task(
        description="make a report",
        expected_output="a report",
        output_pydantic=Report,
        guardrails=[guardrail],
    )

    result = task.execute_sync(agent=agent)

    assert len(seen) == 1
    assert isinstance(seen[0], Report)
    assert seen[0].score == 9
    assert result.pydantic is not None
