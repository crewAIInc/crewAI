"""Tests that per-execution task counters reset between executions."""

import pytest
from unittest.mock import AsyncMock, Mock

from crewai import Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_types import LLMGuardrailStartedEvent
from crewai.tasks.task_output import TaskOutput


def make_agent() -> Mock:
    agent = Mock()
    agent.role = "test_agent"
    agent.crew = None
    agent.last_messages = []
    return agent


def test_retry_count_resets_between_executions():
    """A guardrail retry in one execution must not leak into the next one."""
    guardrail_state = {"failed": False}

    def guardrail(result: TaskOutput):
        if not guardrail_state["failed"]:
            guardrail_state["failed"] = True
            return (False, "retry once")
        return (True, result.raw)

    agent = make_agent()
    agent.execute_task.side_effect = ["bad", "good", "good"]

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        guardrail_max_retries=2,
    )

    task.execute_sync(agent=agent)
    assert task.retry_count == 1

    task.execute_sync(agent=agent)
    assert task.retry_count == 0


def test_tool_counters_reset_between_executions():
    """Tool usage stats from a previous execution must not accumulate."""
    agent = make_agent()
    agent.execute_task.return_value = "ok"

    task = Task(description="Test task", expected_output="Output")
    task.used_tools = 7
    task.tools_errors = 3
    task.delegations = 2
    task._guardrail_retry_counts = {0: 5}

    task.execute_sync(agent=agent)

    assert task.used_tools == 0
    assert task.tools_errors == 0
    assert task.delegations == 0
    assert task._guardrail_retry_counts == {}


def test_guardrail_event_retry_count_resets_between_executions():
    """The first attempt of a new execution must report retry_count 0."""
    started_retry_counts: list[int] = []

    @crewai_event_bus.on(LLMGuardrailStartedEvent)
    def capture(source, event):
        started_retry_counts.append(event.retry_count)

    try:
        guardrail_state = {"failed": False}

        def guardrail(result: TaskOutput):
            if not guardrail_state["failed"]:
                guardrail_state["failed"] = True
                return (False, "retry once")
            return (True, result.raw)

        agent = make_agent()
        agent.execute_task.side_effect = ["bad", "good", "good"]

        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=guardrail,
            guardrail_max_retries=2,
        )

        task.execute_sync(agent=agent)
        crewai_event_bus.flush(timeout=5.0)
        first_run_counts = list(started_retry_counts)

        task.execute_sync(agent=agent)
        crewai_event_bus.flush(timeout=5.0)
        second_run_counts = started_retry_counts[len(first_run_counts) :]

        assert sorted(first_run_counts) == [0, 1]
        assert second_run_counts == [0]
    finally:
        crewai_event_bus.off(LLMGuardrailStartedEvent, capture)


@pytest.mark.asyncio
async def test_async_execution_resets_retry_count():
    """The async core must reset counters the same way as the sync one."""
    agent = make_agent()
    agent.aexecute_task = AsyncMock(side_effect=["bad", "good", "good"])

    guardrail_state = {"failed": False}

    def guardrail(result: TaskOutput):
        if not guardrail_state["failed"]:
            guardrail_state["failed"] = True
            return (False, "retry once")
        return (True, result.raw)

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        guardrail_max_retries=2,
    )

    await task.aexecute_sync(agent=agent)
    assert task.retry_count == 1

    await task.aexecute_sync(agent=agent)
    assert task.retry_count == 0
