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
    assert "error" not in result.lower()
    assert "unauthorized" not in result.lower()

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
    assert "cannot delegate this task" in result.lower()
    assert "tech manager" in result.lower()
    assert "communications manager" in result.lower()

def test_delegate_work_without_allowed_agents():
    """Test delegation works normally when no allowed_agents is specified."""
    # Create agents
    manager = Agent(
        role="Manager",
        goal="Manage the team",
        backstory="An experienced manager",
        allow_delegation=True  # No allowed_agents specified
    )
    worker = Agent(
        role="Worker",
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
    manager.llm = MagicMock()
    manager.llm.invoke = MagicMock(return_value=mock_response)
    manager.llm.call = MagicMock(return_value=mock_content)
    worker.llm = MagicMock()
    worker.llm.invoke = MagicMock(return_value=mock_response)
    worker.llm.call = MagicMock(return_value=mock_content)
    
    # Create crew and tool
    crew = Crew(agents=[manager, worker])
    tool = DelegateWorkTool(
        name="Delegate work to coworker",
        description="Tool for delegating work to coworkers",
        agents=[manager, worker],
        agent_id=manager.id
    )
    
    # Test delegation
    result = tool._execute(
        agent_name="Worker",
        task="Complete task",
        context="Important task"
    )
    
    # Verify delegation was allowed
    assert "error" not in result.lower()
    assert "unauthorized" not in result.lower()
