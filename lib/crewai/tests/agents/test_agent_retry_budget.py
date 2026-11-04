"""Regression tests for the per-task Agent retry budget."""

from unittest.mock import patch

import pytest

from crewai import Agent, Task
from crewai.agents.agent_executor import AgentExecutor


def _make_agent(max_retry_limit: int = 1) -> Agent:
    return Agent(
        role="researcher",
        goal="test goal",
        backstory="test backstory",
        max_retry_limit=max_retry_limit,
    )


def test_retry_budget_is_per_task() -> None:
    """Retries spent on one task must not consume another task's budget."""
    agent = _make_agent(max_retry_limit=1)
    first_task = Task(description="first task", expected_output="o1", agent=agent)
    second_task = Task(description="second task", expected_output="o2", agent=agent)
    agent.create_agent_executor(task=first_task)

    with patch.object(AgentExecutor, "invoke") as invoke_mock:
        invoke_mock.side_effect = Exception("transient failure")

        with pytest.raises(Exception, match="transient failure"):
            agent.execute_task(task=first_task)
        assert invoke_mock.call_count == 2  # initial attempt + one retry

        with pytest.raises(Exception, match="transient failure"):
            agent.execute_task(task=second_task)
        # The second task gets its own initial attempt and one retry; without a
        # per-task counter it raises after a single call (3 in total).
        assert invoke_mock.call_count == 4
