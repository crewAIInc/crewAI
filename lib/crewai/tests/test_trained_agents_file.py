"""Test for custom trained_agents_file support."""
import pytest
from unittest.mock import Mock, patch

from crewai.agent.core import Agent
from crewai.crew import Crew
from crewai.task import Task


@patch("crewai.agent.core.CrewTrainingHandler")
def test_agent_use_trained_data_with_custom_file(crew_training_handler):
    """Test that agent uses crew's trained_agents_file when available."""
    task_prompt = "What is 1 + 1?"
    
    # Create a crew with custom trained_agents_file
    crew = Crew(
        trained_agents_file="my_custom_trained.pkl",
        agents=[],
        tasks=[]
    )
    
    agent = Agent(
        role="researcher",
        goal="test goal",
        backstory="test backstory",
    )
    agent.crew = crew
    
    crew_training_handler.return_value.load.return_value = {
        agent.role: {
            "suggestions": [
                "The result of the math operation must be right.",
            ]
        }
    }

    result = agent._use_trained_data(task_prompt=task_prompt)

    # Should use the custom filename from crew
    crew_training_handler.assert_called_with("my_custom_trained.pkl")
    assert "The result of the math operation must be right." in result


@patch("crewai.agent.core.CrewTrainingHandler")
def test_agent_use_trained_data_fallback_to_default(crew_training_handler):
    """Test that agent falls back to default when crew has no custom file."""
    task_prompt = "What is 1 + 1?"
    
    # Create a crew with default trained_agents_file
    crew = Crew(agents=[], tasks=[])
    
    agent = Agent(
        role="researcher",
        goal="test goal",
        backstory="test backstory",
    )
    agent.crew = crew
    
    crew_training_handler.return_value.load.return_value = {}

    agent._use_trained_data(task_prompt=task_prompt)

    # Should use the default filename
    crew_training_handler.assert_called_with("trained_agents_data.pkl")


def test_crew_trained_agents_file_default():
    """Test that Crew has correct default for trained_agents_file."""
    from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE
    
    crew = Crew(agents=[], tasks=[])
    assert crew.trained_agents_file == TRAINED_AGENTS_DATA_FILE


def test_crew_trained_agents_file_custom():
    """Test that Crew accepts custom trained_agents_file."""
    crew = Crew(
        trained_agents_file="custom_file.pkl",
        agents=[],
        tasks=[]
    )
    assert crew.trained_agents_file == "custom_file.pkl"
