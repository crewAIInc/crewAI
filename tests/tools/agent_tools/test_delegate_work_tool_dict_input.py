"""Test delegate work tool with dictionary inputs."""

import pytest

from crewai.agent import Agent
from crewai.tools.agent_tools.agent_tools import AgentTools

researcher = Agent(
    role="researcher",
    goal="make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology",
    allow_delegation=False,
)
tools = AgentTools(agents=[researcher]).tools()
delegate_tool = tools[0]


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work_with_dict_input():
    """Test that the delegate work tool can handle dictionary inputs."""
    task_dict = {
        "description": "share your take on AI Agents",
        "goal": "provide comprehensive analysis"
    }
    context_dict = {
        "background": "I heard you hate them",
        "additional_info": "We need this for a report"
    }
    
    result = delegate_tool.run(
        coworker="researcher",
        task=task_dict,
        context=context_dict,
    )
    
    assert isinstance(result, str)
    assert len(result) > 0
