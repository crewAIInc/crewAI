import pytest
from unittest.mock import MagicMock, patch

from crewai import Agent, Crew, Task
from crewai.utilities.converter import Converter
from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    string: str = Field(description='string needs to be maintained')

def test_pydantic_model_conversion():
    """Test that pydantic model conversion works without causing import errors."""
    
    # Test data
    test_string = '{"string": "test value"}'
    
    # Create a pydantic model directly
    result = ResponseFormat.model_validate_json(test_string)
    
    # Verify the conversion worked
    assert result is not None
    assert hasattr(result, "string")
    assert isinstance(result.string, str)
    assert result.string == "test value"

@patch('crewai.crew.Crew.kickoff')
def test_output_pydantic_with_mocked_crew(mock_kickoff):
    """Test that output_pydantic works properly without causing import errors."""
    
    # Mock the crew kickoff to return a valid response
    mock_result = ResponseFormat(string="mocked result")
    mock_kickoff.return_value = mock_result
    
    # Create a simple agent
    agent = Agent(
        role="Test Agent",
        goal="Test pydantic model output",
        backstory="Testing pydantic output functionality",
        verbose=True
    )
    
    # Create a task with output_pydantic
    task = Task(
        description="Return a simple string",
        expected_output="A simple string",
        agent=agent,
        output_pydantic=ResponseFormat
    )
    
    # Create a crew with the agent and task
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    # Execute the crew (this will use our mock)
    result = crew.kickoff()
    
    # Verify we got a result
    assert result is not None
    
    # Verify the result has a string attribute (as defined in ResponseFormat)
    assert hasattr(result, "string")
    assert isinstance(result.string, str)
    assert result.string == "mocked result"
