"""Test allowed_agents functionality for hierarchical delegation."""

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tools.agent_tools.agent_tools import AgentTools


@pytest.fixture
def agents():
    """Create test agents for delegation testing."""
    manager = Agent(
        role="Manager",
        goal="Manage the team",
        backstory="You are a team manager",
        allow_delegation=True,
    )
    
    researcher = Agent(
        role="Researcher", 
        goal="Research topics",
        backstory="You are a researcher",
        allow_delegation=False,
    )
    
    writer = Agent(
        role="Writer",
        goal="Write content", 
        backstory="You are a writer",
        allow_delegation=False,
    )
    
    analyst = Agent(
        role="Analyst",
        goal="Analyze data",
        backstory="You are an analyst", 
        allow_delegation=False,
    )
    
    return manager, researcher, writer, analyst


def test_allowed_agents_with_role_strings(agents):
    """Test allowed_agents with role strings."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = ["Researcher", "Writer"]
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    assert len(tools) == 2
    delegate_tool = tools[0]
    
    assert len(delegate_tool.agents) == 2
    agent_roles = [agent.role for agent in delegate_tool.agents]
    assert "Researcher" in agent_roles
    assert "Writer" in agent_roles
    assert "Analyst" not in agent_roles


def test_allowed_agents_with_agent_instances(agents):
    """Test allowed_agents with agent instances."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = [researcher, analyst]
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    assert len(tools) == 2
    delegate_tool = tools[0]
    
    assert len(delegate_tool.agents) == 2
    assert researcher in delegate_tool.agents
    assert analyst in delegate_tool.agents
    assert writer not in delegate_tool.agents


def test_allowed_agents_mixed_types(agents):
    """Test allowed_agents with mixed role strings and agent instances."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = ["Researcher", writer]
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    delegate_tool = tools[0]
    assert len(delegate_tool.agents) == 2
    assert researcher in delegate_tool.agents
    assert writer in delegate_tool.agents
    assert analyst not in delegate_tool.agents


def test_allowed_agents_empty_list(agents):
    """Test allowed_agents with empty list (no delegation allowed)."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = []
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    assert len(tools) == 0


def test_allowed_agents_none(agents):
    """Test allowed_agents with None (delegate to all agents)."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = None
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    delegate_tool = tools[0]
    assert len(delegate_tool.agents) == 3
    assert researcher in delegate_tool.agents
    assert writer in delegate_tool.agents
    assert analyst in delegate_tool.agents


def test_allowed_agents_case_insensitive_matching(agents):
    """Test that role matching is case-insensitive."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = ["researcher", "WRITER"]
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    delegate_tool = tools[0]
    assert len(delegate_tool.agents) == 2
    assert researcher in delegate_tool.agents
    assert writer in delegate_tool.agents


def test_allowed_agents_validation():
    """Test validation of allowed_agents field."""
    agent = Agent(
        role="Test",
        goal="Test",
        backstory="Test",
        allowed_agents=["Role1", "Role2"]
    )
    assert agent.allowed_agents == ["Role1", "Role2"]
    
    agent = Agent(
        role="Test",
        goal="Test", 
        backstory="Test",
        allowed_agents=None
    )
    assert agent.allowed_agents is None
    
    with pytest.raises(ValueError, match="allowed_agents must be a list or tuple of agent roles"):
        Agent(
            role="Test",
            goal="Test",
            backstory="Test",
            allowed_agents="invalid"
        )
    
    with pytest.raises(ValueError, match="Each item in allowed_agents must be either a string"):
        Agent(
            role="Test",
            goal="Test",
            backstory="Test",
            allowed_agents=[123]
        )


def test_crew_integration_with_allowed_agents(agents):
    """Test integration with Crew class."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = ["Researcher"]
    
    task = Task(
        description="Research AI trends",
        expected_output="Research report",
        agent=manager
    )
    
    crew = Crew(
        agents=[manager, researcher, writer, analyst],
        tasks=[task]
    )
    
    tools = crew._prepare_tools(manager, task, [])
    
    delegation_tools = [tool for tool in tools if "Delegate work" in tool.name or "Ask question" in tool.name]
    
    if delegation_tools:
        delegate_tool = next(tool for tool in delegation_tools if "Delegate work" in tool.name)
        assert len(delegate_tool.agents) == 1
        assert delegate_tool.agents[0].role == "Researcher"


def test_backward_compatibility_no_allowed_agents(agents):
    """Test that agents without allowed_agents work as before."""
    manager, researcher, writer, analyst = agents
    
    assert manager.allowed_agents is None
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    assert len(tools) == 2
    delegate_tool = tools[0]
    assert len(delegate_tool.agents) == 3
    assert researcher in delegate_tool.agents
    assert writer in delegate_tool.agents
    assert analyst in delegate_tool.agents


def test_no_delegating_agent_parameter(agents):
    """Test AgentTools.tools() without delegating_agent parameter."""
    manager, researcher, writer, analyst = agents
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools()  # No delegating_agent parameter
    
    assert len(tools) == 2
    delegate_tool = tools[0]
    assert len(delegate_tool.agents) == 3


def test_allowed_agents_with_nonexistent_role(agents):
    """Test allowed_agents with role that doesn't exist in available agents."""
    manager, researcher, writer, analyst = agents
    
    manager.allowed_agents = ["Researcher", "NonExistentRole"]
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    delegate_tool = tools[0]
    assert len(delegate_tool.agents) == 1
    assert researcher in delegate_tool.agents


def test_allowed_agents_with_nonexistent_instance(agents):
    """Test allowed_agents with agent instance that doesn't exist in available agents."""
    manager, researcher, writer, analyst = agents
    
    other_agent = Agent(
        role="Other",
        goal="Other goal",
        backstory="Other backstory"
    )
    
    manager.allowed_agents = [researcher, other_agent]
    
    agent_tools = AgentTools(agents=[researcher, writer, analyst])
    tools = agent_tools.tools(delegating_agent=manager)
    
    delegate_tool = tools[0]
    assert len(delegate_tool.agents) == 1
    assert researcher in delegate_tool.agents
    assert other_agent not in delegate_tool.agents
