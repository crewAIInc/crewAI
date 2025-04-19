from unittest.mock import MagicMock

from crewai import Task
from crewai.utilities.planning_handler import CrewPlanner

def test_planning_llm_inherits_auth_params():
    """Test that planning LLM inherits authentication parameters from agent LLM."""
    mock_llm = MagicMock()
    mock_llm.base_url = "https://api.custom-provider.com/v1"
    mock_llm.api_version = "2023-05-15"
    
    task = Task(
        description="Test Task",
        expected_output="Test Output"
    )
    
    planner = CrewPlanner(
        tasks=[task],
        planning_agent_llm=None,  # This should trigger the inheritance logic
        agent_llm=mock_llm
    )
    
    assert hasattr(planner, 'planning_agent_llm')
    assert hasattr(planner.planning_agent_llm, 'base_url')
    assert planner.planning_agent_llm.base_url == "https://api.custom-provider.com/v1"
    assert planner.planning_agent_llm.api_version == "2023-05-15"
