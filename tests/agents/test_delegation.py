import pytest
from unittest.mock import MagicMock

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool

def test_delegate_work_with_allowed_agents():
    """Test successful delegation to allowed agent."""
    # Create agents
    executive = Agent(
        role="Executive Director",
        goal="Manage the team",
        backstory="An experienced manager",
        allow_delegation=True,
        allowed_agents=["Communications Manager"]
    )
    comms_manager = Agent(
        role="Communications Manager",
        goal="Handle communications",
        backstory="A skilled communicator",
        allow_delegation=False
    )
    
    # Mock LLM to avoid actual API calls
    mock_content = "Thought: I will handle this task\nFinal Answer: Task completed successfully"
    mock_response = {
        "choices": [{
            "message": {
                "content": mock_content
            }
        }]
    }
    executive.llm = MagicMock()
    executive.llm.invoke = MagicMock(return_value=mock_response)
    executive.llm.call = MagicMock(return_value=mock_content)
    comms_manager.llm = MagicMock()
    comms_manager.llm.invoke = MagicMock(return_value=mock_response)
    comms_manager.llm.call = MagicMock(return_value=mock_content)
    
    # Create crew and tool
    crew = Crew(agents=[executive, comms_manager])
    tool = DelegateWorkTool(
        name="Delegate work to coworker",
        description="Tool for delegating work to coworkers",
        agents=[executive, comms_manager],
        agent_id=executive.id
    )
    
    # Test delegation
    result = tool._execute(
        agent_name="Communications Manager",
        task="Write a press release",
        context="Important company announcement"
    )
    
    # Verify delegation was allowed
    assert "authorization error" not in result.lower()
    assert "cannot delegate" not in result.lower()

def test_delegate_work_with_unauthorized_agent():
    """Test failed delegation to unauthorized agent."""
    # Create agents
    executive = Agent(
        role="Executive Director",
        goal="Manage the team",
        backstory="An experienced manager",
        allow_delegation=True,
        allowed_agents=["Communications Manager"]
    )
    tech_manager = Agent(
        role="Tech Manager",
        goal="Manage technology",
        backstory="A tech expert",
        allow_delegation=False
    )
    
    # Mock LLM to avoid actual API calls
    mock_content = "Thought: I will handle this task\nFinal Answer: Task completed successfully"
    mock_response = {
        "choices": [{
            "message": {
                "content": mock_content
            }
        }]
    }
    executive.llm = MagicMock()
    executive.llm.invoke = MagicMock(return_value=mock_response)
    executive.llm.call = MagicMock(return_value=mock_content)
    tech_manager.llm = MagicMock()
    tech_manager.llm.invoke = MagicMock(return_value=mock_response)
    tech_manager.llm.call = MagicMock(return_value=mock_content)
    
    # Create crew and tool
    crew = Crew(agents=[executive, tech_manager])
    tool = DelegateWorkTool(
        name="Delegate work to coworker",
        description="Tool for delegating work to coworkers",
        agents=[executive, tech_manager],
        agent_id=executive.id
    )
    
    # Test delegation
    result = tool._execute(
        agent_name="Tech Manager",
        task="Update servers",
        context="Server maintenance needed"
    )
    
    # Verify delegation was blocked with proper error message
    assert "authorization error" in result.lower()
    assert "tech manager" in result.lower()
    assert "communications manager" in result.lower()

@pytest.mark.parametrize("scenario", [
    {
        "name": "empty_allowed_agents",
        "delegating_agent": {
            "role": "Manager",
            "allow_delegation": True,
            "allowed_agents": []
        },
        "target_agent": "Worker",
        "should_succeed": False,
        "error_contains": "cannot be empty"
    },
    {
        "name": "case_insensitive_match",
        "delegating_agent": {
            "role": "Manager",
            "allow_delegation": True,
            "allowed_agents": ["Worker"]
        },
        "target_agent": "WORKER",
        "should_succeed": True
    },
    {
        "name": "unauthorized_delegation",
        "delegating_agent": {
            "role": "Manager",
            "allow_delegation": True,
            "allowed_agents": ["Worker A"]
        },
        "target_agent": "Worker B",
        "should_succeed": False,
        "error_contains": "Authorization Error"
    },
    {
        "name": "no_allowed_agents_specified",
        "delegating_agent": {
            "role": "Manager",
            "allow_delegation": True,
            "allowed_agents": None
        },
        "target_agent": "Worker",
        "should_succeed": True
    }
])
def test_delegation_scenarios(scenario):
    """Test various delegation scenarios."""
    # Create agents
    delegating_agent = Agent(
        role=scenario["delegating_agent"]["role"],
        goal="Manage the team",
        backstory="An experienced manager",
        allow_delegation=scenario["delegating_agent"]["allow_delegation"],
        allowed_agents=scenario["delegating_agent"]["allowed_agents"]
    )
    target_agent = Agent(
        role=scenario["target_agent"],
        goal="Do the work",
        backstory="A skilled worker",
        allow_delegation=False
    )
    
    # Mock LLM to avoid actual API calls
    mock_content = "Thought: I will handle this task\nFinal Answer: Task completed successfully"
    mock_response = {
        "choices": [{
            "message": {
                "content": mock_content
            }
        }]
    }
    for agent in [delegating_agent, target_agent]:
        agent.llm = MagicMock()
        agent.llm.invoke = MagicMock(return_value=mock_response)
        agent.llm.call = MagicMock(return_value=mock_content)
    
    # Create crew and tool
    crew = Crew(agents=[delegating_agent, target_agent])
    tool = DelegateWorkTool(
        name="Delegate work to coworker",
        description="Tool for delegating work to coworkers",
        agents=[delegating_agent, target_agent],
        agent_id=delegating_agent.id
    )
    
    # Test delegation
    result = tool._execute(
        agent_name=scenario["target_agent"],
        task="Complete task",
        context="Important task"
    )
    
    # Verify results
    if scenario["should_succeed"]:
        assert "authorization error" not in result.lower()
        assert "cannot delegate" not in result.lower()
    else:
        assert scenario["error_contains"].lower() in result.lower()
