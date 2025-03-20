import pytest
from unittest.mock import patch, MagicMock

from crewai.task import Task


def test_task_custom_human_input_parameter():
    """Test that the Task class accepts the ask_human_input parameter."""
    # Custom human input function
    def custom_input_func(final_answer):
        return "Custom feedback"
    
    # Create a task with the custom function
    task = Task(
        description="Test task",
        expected_output="Test output",
        human_input=True,
        ask_human_input=custom_input_func
    )
    
    # Verify the parameter was stored correctly
    assert task.ask_human_input == custom_input_func
    assert callable(task.ask_human_input)
