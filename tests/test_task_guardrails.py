"""Tests for task guardrails functionality."""

from unittest.mock import Mock

import pytest

from crewai.task import Task
from crewai.tasks.task_output import TaskOutput


def test_task_without_guardrail():
    """Test that tasks work normally without guardrails (backward compatibility)."""
    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "test result"
    agent.crew = None

    task = Task(description="Test task", expected_output="Output")

    result = task.execute_sync(agent=agent)
    assert isinstance(result, TaskOutput)
    assert result.raw == "test result"


def test_task_with_successful_guardrail():
    """Test that successful guardrail validation passes transformed result."""

    def guardrail(result: TaskOutput):
        return (True, result.raw.upper())

    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "test result"
    agent.crew = None

    task = Task(description="Test task", expected_output="Output", guardrail=guardrail)

    result = task.execute_sync(agent=agent)
    assert isinstance(result, TaskOutput)
    assert result.raw == "TEST RESULT"


def test_task_with_failing_guardrail():
    """Test that failing guardrail triggers retry with error context."""

    def guardrail(result: TaskOutput):
        return (False, "Invalid format")

    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.side_effect = ["bad result", "good result"]
    agent.crew = None

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        max_retries=1,
    )

    # First execution fails guardrail, second succeeds
    agent.execute_task.side_effect = ["bad result", "good result"]
    with pytest.raises(Exception) as exc_info:
        task.execute_sync(agent=agent)

    assert "Task failed guardrail validation" in str(exc_info.value)
    assert task.retry_count == 1


def test_task_with_guardrail_retries():
    """Test that guardrail respects max_retries configuration."""

    def guardrail(result: TaskOutput):
        return (False, "Invalid format")

    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "bad result"
    agent.crew = None

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        max_retries=2,
    )

    with pytest.raises(Exception) as exc_info:
        task.execute_sync(agent=agent)

    assert task.retry_count == 2
    assert "Task failed guardrail validation after 2 retries" in str(exc_info.value)
    assert "Invalid format" in str(exc_info.value)


def test_guardrail_error_in_context():
    """Test that guardrail error is passed in context for retry."""

    def guardrail(result: TaskOutput):
        return (False, "Expected JSON, got string")

    agent = Mock()
    agent.role = "test_agent"
    agent.crew = None

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        max_retries=1,
    )

    # Mock execute_task to succeed on second attempt
    first_call = True

    def execute_task(task, context, tools):
        nonlocal first_call
        if first_call:
            first_call = False
            return "invalid"
        return '{"valid": "json"}'

    agent.execute_task.side_effect = execute_task

    with pytest.raises(Exception) as exc_info:
        task.execute_sync(agent=agent)

    assert "Task failed guardrail validation" in str(exc_info.value)
    assert "Expected JSON, got string" in str(exc_info.value)
