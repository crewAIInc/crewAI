from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.base import BaseLanguageModel

from crewai import Agent, Crew, Process, Task


def test_hierarchical_delegation_tool_availability():
    """Test that all agents are available for delegation in hierarchical mode."""
    # Mock LLM to avoid actual API calls
    mock_llm = MagicMock(spec=BaseLanguageModel)
    
    # Create agents
    manager = Agent(
        role="Manager",
        goal="Manage the team",
        backstory="I am a manager",
        allow_delegation=True,
        llm=mock_llm
    )
    
    kb_agent = Agent(
        role="kb_retriever_agent",
        goal="Retrieve knowledge",
        backstory="I am a knowledge retrieval specialist",
        allow_delegation=True,
        llm=mock_llm
    )
    
    worker = Agent(
        role="Worker",
        goal="Do the work",
        backstory="I am a worker",
        allow_delegation=True,
        llm=mock_llm
    )
    
    # Create a task assigned to the manager
    task = Task(
        description="Complex task requiring delegation",
        expected_output="Task completion status",
        agent=manager
    )
    
    # Create the crew with hierarchical process
    crew = Crew(
        agents=[kb_agent, worker],  # Manager should not be in agents list when using hierarchical process
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=manager  # Explicitly set the manager agent for hierarchical process
    )
    
    # Get the manager's tools
    tools_for_task = task.tools or manager.tools or []
    manager_tools = crew._prepare_tools(manager, task, tools_for_task)
    
    # Find delegation tools
    delegation_tools = [tool for tool in manager_tools if tool.name == "Delegate work to coworker"]
    assert len(delegation_tools) > 0, "Delegation tool should be present"
    
    # Get the delegation tool description
    delegate_tool = delegation_tools[0]
    tool_description = str(delegate_tool.description)
    
    # Verify all agents are available for delegation
    assert "kb_retriever_agent" in tool_description, "kb_retriever_agent should be available for delegation"
    assert "Worker" in tool_description, "Worker should be available for delegation"

def test_hierarchical_delegation_tool_updates():
    """Test that delegation tools are properly updated when task agent changes."""
    mock_llm = MagicMock(spec=BaseLanguageModel)
    
    manager = Agent(
        role="Manager",
        goal="Manage the team",
        backstory="I am a manager",
        allow_delegation=True,
        llm=mock_llm
    )
    
    kb_agent = Agent(
        role="kb_retriever_agent",
        goal="Retrieve knowledge",
        backstory="I am a knowledge retrieval specialist",
        allow_delegation=True,
        llm=mock_llm
    )
    
    # Create tasks for different agents
    manager_task = Task(
        description="Manager task",
        expected_output="Manager task completion status",
        agent=manager
    )
    
    kb_task = Task(
        description="KB task",
        expected_output="KB task completion status",
        agent=kb_agent
    )
    
    # Create crew
    crew = Crew(
        agents=[kb_agent],  # Manager should not be in agents list when using hierarchical process
        tasks=[manager_task, kb_task],
        process=Process.hierarchical,
        manager_agent=manager  # Explicitly set the manager agent for hierarchical process
    )
    
    # Test manager's tools
    manager_tools_for_task = manager_task.tools or manager.tools or []
    manager_tools = crew._prepare_tools(manager, manager_task, manager_tools_for_task)
    manager_delegation = [t for t in manager_tools if t.name == "Delegate work to coworker"]
    assert len(manager_delegation) > 0, "Manager should have delegation tool"
    assert "kb_retriever_agent" in str(manager_delegation[0].description), "Manager should see kb_retriever_agent"
    
    # Test kb_agent's tools
    kb_tools_for_task = kb_task.tools or kb_agent.tools or []
    kb_tools = crew._prepare_tools(kb_agent, kb_task, kb_tools_for_task)
    kb_delegation = [t for t in kb_tools if t.name == "Delegate work to coworker"]
    assert len(kb_delegation) > 0, "KB agent should have delegation tool"
    assert "Manager" in str(kb_delegation[0].description), "KB agent should see manager"
