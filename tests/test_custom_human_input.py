from unittest.mock import MagicMock, patch

import pytest

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


def test_task_invalid_human_input_parameter():
    """Test that non-callable input raises validation error."""
    with pytest.raises(ValueError) as exc_info:
        Task(
            description="Test task",
            expected_output="Test output",
            human_input=True,
            ask_human_input="not_a_function"
        )
    
    assert "Input should be callable" in str(exc_info.value)


def test_custom_input_function_error_handling():
    """Test handling of errors in custom input function."""
    def failing_input(_):
        raise Exception("API Error")
    
    # Create a simplified test for error handling
    # We'll directly test the error handling in the _handle_human_feedback method
    
    # Create a mock agent finish object with a simple output
    agent_finish = MagicMock()
    agent_finish.output = "Test output"
    
    # Create a mock executor with our failing function
    executor = MagicMock()
    executor.ask_human_input_function = failing_input
    
    # Set up the default input method mock
    executor._ask_human_input = MagicMock(return_value="Default input used")
    
    # Add the extract method that returns the output directly
    executor._extract_output_from_agent_finish = MagicMock(return_value="Test output")
    
    # Test the error handling by calling the method directly
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    
    # Capture print output to verify error message
    with patch('builtins.print') as mock_print:
        # Call the method we're testing
        CrewAgentExecutor._handle_human_feedback(executor, agent_finish)
        
        # Verify error was printed
        mock_print.assert_called_once()
        assert "Error using custom input function" in mock_print.call_args[0][0]
        
    # Verify fallback to default method occurred
    executor._ask_human_input.assert_called_once_with("Test output")
