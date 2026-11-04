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


def test_retry_budget_is_not_reset_by_another_task_starting() -> None:
    """Another task's prepare/finalize must not extend an in-flight retry chain."""
    agent = _make_agent(max_retry_limit=1)
    first_task = Task(description="first task", expected_output="o1", agent=agent)
    second_task = Task(description="second task", expected_output="o2", agent=agent)
    agent.create_agent_executor(task=first_task)

    calls = 0

    def invoke_side_effect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            # A concurrent task on the same agent enters and leaves while the
            # first task is between its attempt and its retry.
            agent._prepare_task_execution(second_task, None)
            agent._finalize_task_execution(second_task, "done")
        raise Exception("transient failure")

    with patch.object(AgentExecutor, "invoke", side_effect=invoke_side_effect):
        with pytest.raises(Exception, match="transient failure"):
            agent.execute_task(task=first_task)

    # Initial attempt plus the one retry ``max_retry_limit=1`` allows. Resetting
    # the budget from the other task's lifecycle would let a third attempt run.
    assert calls == 2
