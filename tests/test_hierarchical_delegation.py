import pytest
from unittest.mock import MagicMock, patch
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.process import Process
from crewai.tools.agent_tools.agent_tools import AgentTools

def test_manager_delegation_with_task_agent():
    """Test that manager can delegate to all agents even when task.agent is specified."""
    # Create mock LLM
    mock_llm = MagicMock()
    
    # Create agents
    manager = Agent(
        role="Manager",
        goal="Manage the team",
        backstory="I am the manager",
        allow_delegation=True,
        llm=mock_llm,
    )
    
    agent1 = Agent(
        role="Agent1",
        goal="Do task 1",
        backstory="I am agent 1",
        llm=mock_llm,
    )
    
    agent2 = Agent(
        role="Agent2",
        goal="Do task 2",
        backstory="I am agent 2",
        llm=mock_llm,
    )
    
    # Create task with specific agent
    task = Task(
        description="Test task",
        expected_output="Test output",  # Required field
        agent=agent1
    )
    
    # Create crew in hierarchical mode
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task],
        manager_agent=manager,
        process=Process.hierarchical,
    )
    
    # Get tools that would be available to the manager
    tools = crew._update_manager_tools(task, [])
    
    # Extract delegation tool
    from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool
    delegation_tools = [t for t in tools if isinstance(t, DelegateWorkTool)]
    assert len(delegation_tools) > 0, "No delegation tools found"
    
    # Verify all agents are available for delegation
    delegation_tool = delegation_tools[0]
    available_agents = delegation_tool.agents
    
    # Manager should see both agents as delegation targets
    assert len(available_agents) == 2, "Manager should see all agents"
    assert agent1 in available_agents, "Agent1 should be available for delegation"
    assert agent2 in available_agents, "Agent2 should be available for delegation"

def test_manager_delegation_without_task_agent():
    """Test that manager can delegate to all agents when no task.agent is specified."""
    # Create mock LLM
    mock_llm = MagicMock()
    
    # Create agents
    manager = Agent(
        role="Manager",
        goal="Manage the team",
        backstory="I am the manager",
        allow_delegation=True,
        llm=mock_llm,
    )
    
    agent1 = Agent(
        role="Agent1",
        goal="Do task 1",
        backstory="I am agent 1",
        llm=mock_llm,
    )
    
    agent2 = Agent(
        role="Agent2",
        goal="Do task 2",
        backstory="I am agent 2",
        llm=mock_llm,
    )
    
    # Create task without specific agent
    task = Task(
        description="Test task",
        expected_output="Test output"  # Required field
    )
    
    # Create crew in hierarchical mode
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task],
        manager_agent=manager,
        process=Process.hierarchical,
    )
    
    # Get tools that would be available to the manager
    tools = crew._update_manager_tools(task, [])
    
    # Extract delegation tool
    from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool
    delegation_tools = [t for t in tools if isinstance(t, DelegateWorkTool)]
    assert len(delegation_tools) > 0, "No delegation tools found"
    
    # Verify all agents are available for delegation
    delegation_tool = delegation_tools[0]
    available_agents = delegation_tool.agents
    
    # Manager should see both agents as delegation targets
    assert len(available_agents) == 2, "Manager should see all agents"
    assert agent1 in available_agents, "Agent1 should be available for delegation"
    assert agent2 in available_agents, "Agent2 should be available for delegation"
