import pytest
from unittest.mock import patch, MagicMock
from crewai import Agent, Task, Crew


@patch('crewai_tools.CodeInterpreterTool')
def test_crew_with_custom_execution_image_integration(mock_code_interpreter_class):
    """Integration test for custom execution image in a Crew workflow."""
    mock_tool_instance = MagicMock()
    mock_code_interpreter_class.return_value = mock_tool_instance
    
    agent = Agent(
        role="Python Developer",
        goal="Execute Python code",
        backstory="Expert in Python programming",
        allow_code_execution=True,
        execution_image="python:3.11-slim"
    )
    
    task = Task(
        description="Calculate 2 + 2",
        expected_output="The result of 2 + 2",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task]
    )
    
    tools = crew._prepare_tools(task, agent)
    
    mock_code_interpreter_class.assert_called_with(
        unsafe_mode=False,
        default_image_tag="python:3.11-slim"
    )
