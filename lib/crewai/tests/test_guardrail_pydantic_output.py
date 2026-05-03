"""Tests for guardrail pydantic output consistency (issue #4369).

This test file verifies that TaskOutput.pydantic is consistently populated
on both the first guardrail invocation and subsequent retry attempts.
"""

from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput


class OutputModel(BaseModel):
    """Test Pydantic model for task output."""

    message: str
    status: str


def create_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.role = "test_agent"
    agent.crew = None
    agent.last_messages = []
    agent.verbose = False
    return agent


def test_guardrail_receives_pydantic_output_on_first_attempt():
    """Test that TaskOutput.pydantic is populated on the first guardrail invocation."""
    # Track guardrail invocations
    invocations = []

    def my_guardrail(task_output: TaskOutput) -> tuple[bool, TaskOutput | str]:
        invocations.append({
            "pydantic": task_output.pydantic,
            "raw": task_output.raw,
            "is_none": task_output.pydantic is None,
        })
        # Accept on first attempt to complete the task
        return (True, task_output)

    agent = create_agent()
    # Return a valid JSON that matches OutputModel schema
    agent.execute_task.return_value = '{"message": "Hello", "status": "success"}'

    task = Task(
        description="Test task",
        expected_output="JSON with message and status",
        output_pydantic=OutputModel,
        guardrail=my_guardrail,
        agent=Agent(role="test", goal="test", backstory="test"),
    )

    result = task.execute_sync(agent=agent)

    # Verify guardrail was called once
    assert len(invocations) == 1

    # CRITICAL: pydantic should NOT be None on first attempt
    assert invocations[0]["pydantic"] is not None, (
        "TaskOutput.pydantic should be populated on first guardrail invocation"
    )
    assert isinstance(invocations[0]["pydantic"], OutputModel)
    assert invocations[0]["pydantic"].message == "Hello"
    assert invocations[0]["pydantic"].status == "success"


def test_guardrail_receives_pydantic_output_on_retry():
    """Test that TaskOutput.pydantic is populated on guardrail retry attempts."""
    # Track guardrail invocations
    invocations = []

    def my_guardrail(task_output: TaskOutput) -> tuple[bool, TaskOutput | str]:
        invocations.append({
            "attempt": len(invocations) + 1,
            "pydantic": task_output.pydantic,
            "is_none": task_output.pydantic is None,
        })

        # Fail first attempt, succeed on second
        if len(invocations) == 1:
            return (False, "Retry for testing")
        return (True, task_output)

    agent = create_agent()
    # Return valid JSON on both attempts
    agent.execute_task.side_effect = [
        '{"message": "First", "status": "fail"}',
        '{"message": "Second", "status": "success"}',
    ]

    task = Task(
        description="Test task",
        expected_output="JSON with message and status",
        output_pydantic=OutputModel,
        guardrail=my_guardrail,
        guardrail_max_retries=1,
        agent=Agent(role="test", goal="test", backstory="test"),
    )

    result = task.execute_sync(agent=agent)

    # Verify guardrail was called twice (first attempt + 1 retry)
    assert len(invocations) == 2

    # CRITICAL: Both attempts should have pydantic populated
    for i, invocation in enumerate(invocations):
        assert invocation["pydantic"] is not None, (
            f"TaskOutput.pydantic should be populated on attempt {i+1}"
        )
        assert isinstance(invocation["pydantic"], OutputModel)


def test_pydantic_consistency_across_attempts():
    """Test that pydantic output structure is consistent across all guardrail attempts."""
    # Track all pydantic outputs
    pydantic_outputs = []

    def my_guardrail(task_output: TaskOutput) -> tuple[bool, TaskOutput | str]:
        pydantic_outputs.append(task_output.pydantic)

        # Fail first two attempts, succeed on third
        if len(pydantic_outputs) < 3:
            return (False, "Keep retrying")
        return (True, task_output)

    agent = create_agent()
    # Return valid JSON on all attempts
    agent.execute_task.side_effect = [
        '{"message": "Attempt 1", "status": "retry"}',
        '{"message": "Attempt 2", "status": "retry"}',
        '{"message": "Attempt 3", "status": "success"}',
    ]

    task = Task(
        description="Test task",
        expected_output="JSON with message and status",
        output_pydantic=OutputModel,
        guardrail=my_guardrail,
        guardrail_max_retries=2,
        agent=Agent(role="test", goal="test", backstory="test"),
    )

    result = task.execute_sync(agent=agent)

    # Verify all three attempts had pydantic output
    assert len(pydantic_outputs) == 3

    # ALL attempts should have pydantic populated (not None)
    for i, pydantic_output in enumerate(pydantic_outputs):
        assert pydantic_output is not None, (
            f"Attempt {i+1} should have pydantic output"
        )
        assert isinstance(pydantic_output, OutputModel)


def test_guardrail_can_validate_pydantic_fields():
    """Test that guardrails can reliably validate pydantic model fields on first attempt."""

    def status_guardrail(task_output: TaskOutput) -> tuple[bool, TaskOutput | str]:
        # This should work on the first attempt (issue #4369 would cause this to fail)
        if task_output.pydantic is None:
            return (False, "Pydantic output is None - cannot validate!")

        # Validate the status field
        if task_output.pydantic.status != "success":
            return (False, "Status must be 'success'")

        return (True, task_output)

    agent = create_agent()
    # First attempt returns invalid status, second returns valid
    agent.execute_task.side_effect = [
        '{"message": "Test", "status": "pending"}',
        '{"message": "Test", "status": "success"}',
    ]

    task = Task(
        description="Test task",
        expected_output="JSON with success status",
        output_pydantic=OutputModel,
        guardrail=status_guardrail,
        guardrail_max_retries=1,
        agent=Agent(role="test", goal="test", backstory="test"),
    )

    result = task.execute_sync(agent=agent)

    # Verify the task succeeded after retry
    assert result.pydantic.status == "success"


@pytest.mark.asyncio
async def test_async_guardrail_receives_pydantic_output():
    """Test that async execution also populates pydantic on first attempt."""
    from unittest.mock import AsyncMock

    # Track guardrail invocations
    invocations = []

    def my_guardrail(task_output: TaskOutput) -> tuple[bool, TaskOutput | str]:
        invocations.append({
            "pydantic": task_output.pydantic,
            "is_none": task_output.pydantic is None,
        })
        return (True, task_output)

    agent = create_agent()
    # Mock async execution - needs to be AsyncMock
    async def mock_aexecute_task(*args, **kwargs):
        return '{"message": "Async test", "status": "success"}'

    agent.aexecute_task = mock_aexecute_task

    task = Task(
        description="Async test task",
        expected_output="JSON with message and status",
        output_pydantic=OutputModel,
        guardrail=my_guardrail,
        agent=Agent(role="test", goal="test", backstory="test"),
    )

    result = await task.aexecute_sync(agent=agent)

    # Verify guardrail was called
    assert len(invocations) == 1

    # CRITICAL: pydantic should be populated in async execution too
    assert invocations[0]["pydantic"] is not None, (
        "TaskOutput.pydantic should be populated in async execution"
    )
    assert isinstance(invocations[0]["pydantic"], OutputModel)
