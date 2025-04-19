from crewai import Agent, Task
from crewai.llm import LLM
from crewai.utilities.planning_handler import CrewPlanner

def test_planning_llm_inherits_auth_params():
    """Test that planning LLM inherits authentication parameters from agent LLM."""
    custom_llm = LLM(
        model="custom-model",
        base_url="https://api.custom-provider.com/v1",
        api_key="fake-api-key",
        api_version="2023-05-15",
        organization="custom-org"
    )
    
    agent = Agent(
        role="Test Agent",
        goal="Test Goal",
        backstory="Test Backstory",
        llm=custom_llm
    )
    
    task = Task(
        description="Test Task",
        expected_output="Test Output",
        agent=agent
    )
    
    planner = CrewPlanner(
        tasks=[task],
        planning_agent_llm=None,  # This should trigger the inheritance logic
        agent_llm=custom_llm
    )
    
    assert hasattr(planner, 'planning_agent_llm')
    assert hasattr(planner.planning_agent_llm, 'base_url')
    assert planner.planning_agent_llm.base_url == "https://api.custom-provider.com/v1"
    assert planner.planning_agent_llm.api_key == "fake-api-key"
    assert planner.planning_agent_llm.api_version == "2023-05-15"
    # organization is not directly accessible as an attribute
