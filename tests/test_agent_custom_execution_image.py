import pytest
from unittest.mock import patch
from crewai import Agent
from pydantic import ValidationError


def test_agent_with_custom_execution_image():
    """Test that Agent can be created with custom execution image."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        allow_code_execution=True,
        execution_image="my-custom-image:latest"
    )
    
    assert agent.execution_image == "my-custom-image:latest"


def test_agent_without_custom_execution_image():
    """Test that Agent works without custom execution image (default behavior)."""
    agent = Agent(
        role="Test Agent", 
        goal="Test goal",
        backstory="Test backstory",
        allow_code_execution=True
    )
    
    assert agent.execution_image is None


@patch('crewai_tools.CodeInterpreterTool')
def test_get_code_execution_tools_with_custom_image(mock_code_interpreter):
    """Test that get_code_execution_tools passes custom image to CodeInterpreterTool."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal", 
        backstory="Test backstory",
        allow_code_execution=True,
        execution_image="my-custom-image:latest"
    )
    
    agent.get_code_execution_tools()
    
    mock_code_interpreter.assert_called_once_with(
        unsafe_mode=False,
        default_image_tag="my-custom-image:latest"
    )


@patch('crewai_tools.CodeInterpreterTool')
def test_get_code_execution_tools_without_custom_image(mock_code_interpreter):
    """Test that get_code_execution_tools works without custom image."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory", 
        allow_code_execution=True
    )
    
    agent.get_code_execution_tools()
    
    mock_code_interpreter.assert_called_once_with(unsafe_mode=False)


@patch('crewai_tools.CodeInterpreterTool')
def test_get_code_execution_tools_with_unsafe_mode_and_custom_image(mock_code_interpreter):
    """Test that both unsafe_mode and custom image work together."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        allow_code_execution=True,
        code_execution_mode="unsafe",
        execution_image="my-custom-image:latest"
    )
    
    agent.get_code_execution_tools()
    
    mock_code_interpreter.assert_called_once_with(
        unsafe_mode=True,
        default_image_tag="my-custom-image:latest"
    )


def test_agent_with_invalid_execution_image():
    """Test that Agent validates execution image format."""
    with pytest.raises(ValidationError, match="String should match pattern"):
        Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            allow_code_execution=True,
            execution_image="invalid@@image"
        )


def test_agent_with_valid_execution_image_formats():
    """Test that Agent accepts various valid Docker image formats."""
    valid_images = [
        "python:3.11",
        "python:3.11-slim",
        "registry.example.com/python:3.11",
        "my-registry.com:5000/python:latest",
        "python",
        "ubuntu:20.04",
        "gcr.io/project/image:tag"
    ]
    
    for image in valid_images:
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            allow_code_execution=True,
            execution_image=image
        )
        assert agent.execution_image == image
