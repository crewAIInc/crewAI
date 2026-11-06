"""Regression tests for per-task Agent tool-result state."""

from crewai import Agent, Task


def _make_agent() -> Agent:
    return Agent(role="researcher", goal="test goal", backstory="test backstory")


def test_prepare_task_execution_clears_previous_tool_results() -> None:
    """A new task must not start with the previous task's tool results."""
    agent = _make_agent()
    task = Task(description="second task", expected_output="answer", agent=agent)
    agent.tools_results = [{"result": "STALE", "result_as_answer": True}]

    agent._prepare_task_execution(task, None)

    assert agent.tools_results == []


def test_stale_result_as_answer_does_not_override_next_task() -> None:
    """``result_as_answer`` from an earlier task must not become this answer."""
    agent = _make_agent()
    task = Task(description="second task", expected_output="answer", agent=agent)
    agent.tools_results = [{"result": "STALE", "result_as_answer": True}]

    agent._prepare_task_execution(task, None)
    result = agent._finalize_task_execution(task, "FRESH")

    assert result == "FRESH"
