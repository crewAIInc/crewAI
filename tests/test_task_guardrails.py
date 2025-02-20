"""Tests for task guardrails functionality."""

from typing import Dict, Any
from unittest.mock import Mock

import pytest

from crewai.task import Task
from crewai.tasks.exceptions import GuardrailValidationError
from crewai.tasks.task_output import TaskOutput


class TestTaskGuardrails:
    """Test suite for task guardrail functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Fixture providing a mock agent for testing."""
        agent = Mock()
        agent.role = "test_agent"
        agent.crew = None
        return agent

    def test_task_without_guardrail(self, mock_agent):
        """Test that tasks work normally without guardrails (backward compatibility)."""
        mock_agent.execute_task.return_value = "test result"
        task = Task(description="Test task", expected_output="Output")

        result = task.execute_sync(agent=mock_agent)
        assert isinstance(result, TaskOutput)
        assert result.raw == "test result"


    def test_task_with_successful_guardrail(self, mock_agent):
        """Test that successful guardrail validation passes transformed result."""
        def guardrail(result: TaskOutput):
            return (True, result.raw.upper())

        mock_agent.execute_task.return_value = "test result"
        task = Task(description="Test task", expected_output="Output", guardrail=guardrail)

        result = task.execute_sync(agent=mock_agent)
        assert isinstance(result, TaskOutput)
        assert result.raw == "TEST RESULT"


    def test_task_with_failing_guardrail(self, mock_agent):
        """Test that failing guardrail triggers retry with error context."""
        def guardrail(result: TaskOutput):
            return (False, "Invalid format")

        mock_agent.execute_task.side_effect = ["bad result", "good result"]
        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=guardrail,
            max_retries=1,
        )

        # First execution fails guardrail, second succeeds
        mock_agent.execute_task.side_effect = ["bad result", "good result"]
        with pytest.raises(Exception) as exc_info:
            task.execute_sync(agent=mock_agent)

        assert "Task failed guardrail validation" in str(exc_info.value)
        assert task.retry_count == 1


    def test_task_with_guardrail_retries(self, mock_agent):
        """Test that guardrail respects max_retries configuration."""
        def guardrail(result: TaskOutput):
            return (False, "Invalid format")

        mock_agent.execute_task.return_value = "bad result"
        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=guardrail,
            max_retries=2,
        )

        with pytest.raises(Exception) as exc_info:
            task.execute_sync(agent=mock_agent)

        assert task.retry_count == 2
        assert "Task failed guardrail validation after 2 retries" in str(exc_info.value)
        assert "Invalid format" in str(exc_info.value)


    def test_guardrail_error_in_context(self, mock_agent):
        """Test that guardrail error is passed in context for retry."""
        def guardrail(result: TaskOutput):
            return (False, "Expected JSON, got string")

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

        mock_agent.execute_task.side_effect = execute_task

        with pytest.raises(Exception) as exc_info:
            task.execute_sync(agent=mock_agent)

        assert "Task failed guardrail validation" in str(exc_info.value)
        assert "Expected JSON, got string" in str(exc_info.value)


    def test_guardrail_with_new_style_annotation(self, mock_agent):
        """Test guardrail with new style tuple annotation."""
        def guardrail(result: TaskOutput) -> tuple[bool, str]:
            return (True, result.raw.upper())
        
        mock_agent.execute_task.return_value = "test result"
        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=guardrail
        )

        result = task.execute_sync(agent=mock_agent)
        assert isinstance(result, TaskOutput)
        assert result.raw == "TEST RESULT"

    def test_guardrail_with_optional_params(self, mock_agent):
        """Test guardrail with optional parameters."""
        def guardrail(result: TaskOutput, optional_param: str = "default") -> tuple[bool, str]:
            return (True, f"{result.raw}-{optional_param}")
        
        mock_agent.execute_task.return_value = "test"
        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=guardrail
        )

        result = task.execute_sync(agent=mock_agent)
        assert isinstance(result, TaskOutput)
        assert result.raw == "test-default"

    def test_guardrail_with_invalid_optional_params(self, mock_agent):
        """Test guardrail with invalid optional parameters."""
        def guardrail(result: TaskOutput, *, required_kwonly: str) -> tuple[bool, str]:
            return (True, result.raw)
        
        with pytest.raises(GuardrailValidationError) as exc_info:
            Task(
                description="Test task",
                expected_output="Output",
                guardrail=guardrail
            )
        assert "exactly one required positional parameter" in str(exc_info.value)

    def test_guardrail_with_dict_return_type(self, mock_agent):
        """Test guardrail with dict return type."""
        def guardrail(result: TaskOutput) -> tuple[bool, dict[str, Any]]:
            return (True, {"processed": result.raw.upper()})
        
        mock_agent.execute_task.return_value = "test"
        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=guardrail
        )

        result = task.execute_sync(agent=mock_agent)
        assert isinstance(result, TaskOutput)
        assert result.raw == {"processed": "TEST"}
