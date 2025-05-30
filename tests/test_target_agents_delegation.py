"""Test target_agents delegation functionality."""

from crewai.agent import Agent

def test_target_agents_filters_delegation_tools():
    """Test that target_agents properly filters available agents for delegation."""
    researcher = Agent(
        role="researcher",
        goal="Research topics",
        backstory="Expert researcher",
        allow_delegation=True,
        target_agents=["writer"]
    )
    
    writer = Agent(
        role="writer", 
        goal="Write content",
        backstory="Expert writer",
        allow_delegation=False
    )
    
    analyst = Agent(
        role="analyst",
        goal="Analyze data", 
        backstory="Expert analyst",
        allow_delegation=False
    )
    
    tools = researcher.get_delegation_tools([writer, analyst])
    delegate_tool = tools[0]
    
    result = delegate_tool.run(
        coworker="writer",
        task="Write an article",
        context="About AI"
    )
    assert "Error executing tool" not in result
    
    result = delegate_tool.run(
        coworker="analyst", 
        task="Analyze data",
        context="Sales data"
    )
    assert "Error executing tool" in result
    assert "analyst" not in result.lower() or "not found" in result.lower()

def test_target_agents_none_allows_all():
    """Test that target_agents=None allows delegation to all agents."""
    researcher = Agent(
        role="researcher",
        goal="Research topics", 
        backstory="Expert researcher",
        allow_delegation=True,
        target_agents=None  # Should allow all
    )
    
    writer = Agent(
        role="writer",
        goal="Write content",
        backstory="Expert writer"
    )
    
    analyst = Agent(
        role="analyst", 
        goal="Analyze data",
        backstory="Expert analyst"
    )
    
    tools = researcher.get_delegation_tools([writer, analyst])
    delegate_tool = tools[0]
    
    result1 = delegate_tool.run(coworker="writer", task="Write", context="test")
    result2 = delegate_tool.run(coworker="analyst", task="Analyze", context="test")
    
    assert "Error executing tool" not in result1
    assert "Error executing tool" not in result2

def test_target_agents_empty_list_blocks_all():
    """Test that target_agents=[] blocks delegation to all agents."""
    researcher = Agent(
        role="researcher",
        goal="Research topics",
        backstory="Expert researcher", 
        allow_delegation=True,
        target_agents=[]  # Should block all
    )
    
    writer = Agent(
        role="writer",
        goal="Write content", 
        backstory="Expert writer"
    )
    
    tools = researcher.get_delegation_tools([writer])
    delegate_tool = tools[0]
    
    result = delegate_tool.run(
        coworker="writer",
        task="Write an article", 
        context="About AI"
    )
    assert "Error executing tool" in result

def test_target_agents_with_invalid_names():
    """Test behavior when target_agents contains invalid agent names."""
    researcher = Agent(
        role="researcher",
        goal="Research topics",
        backstory="Expert researcher",
        allow_delegation=True, 
        target_agents=["writer", "nonexistent_agent"]
    )
    
    writer = Agent(
        role="writer",
        goal="Write content",
        backstory="Expert writer" 
    )
    
    tools = researcher.get_delegation_tools([writer])
    delegate_tool = tools[0]
    
    result = delegate_tool.run(
        coworker="writer",
        task="Write an article",
        context="About AI"
    )
    assert "Error executing tool" not in result
