"""Test MCP imports to ensure issue #3858 is resolved.

This test file specifically addresses GitHub issue #3858 where users
reported ModuleNotFoundError when trying to import MCP classes from crewai.mcp.

The issue was that the documentation showed importing from crewai.mcp, but
these classes were not available in version 1.3.0. This was fixed in version 1.4.0
with the addition of first-class MCP support.
"""

import pytest
from crewai import Agent, Task, Crew
from crewai.mcp import MCPServerStdio, MCPServerHTTP, MCPServerSSE


def test_mcp_classes_can_be_imported():
    """Test that MCP server configuration classes can be imported from crewai.mcp.
    
    This test addresses issue #3858 where users got:
    ModuleNotFoundError: No module named 'crewai.mcp'
    """
    assert MCPServerStdio is not None
    assert MCPServerHTTP is not None
    assert MCPServerSSE is not None


def test_mcp_server_stdio_instantiation():
    """Test that MCPServerStdio can be instantiated as shown in the documentation."""
    mcp_server = MCPServerStdio(
        command="python",
        args=["local_server.py"],
        env={"API_KEY": "your_key"},
    )
    
    assert mcp_server.command == "python"
    assert mcp_server.args == ["local_server.py"]
    assert mcp_server.env == {"API_KEY": "your_key"}


def test_mcp_server_http_instantiation():
    """Test that MCPServerHTTP can be instantiated as shown in the documentation."""
    mcp_server = MCPServerHTTP(
        url="https://api.research.com/mcp",
        headers={"Authorization": "Bearer your_token"},
    )
    
    assert mcp_server.url == "https://api.research.com/mcp"
    assert mcp_server.headers == {"Authorization": "Bearer your_token"}


def test_mcp_server_sse_instantiation():
    """Test that MCPServerSSE can be instantiated."""
    mcp_server = MCPServerSSE(
        url="https://api.example.com/mcp/sse",
        headers={"Authorization": "Bearer your_token"},
    )
    
    assert mcp_server.url == "https://api.example.com/mcp/sse"
    assert mcp_server.headers == {"Authorization": "Bearer your_token"}


def test_agent_accepts_mcp_configs_as_documented():
    """Test that Agent accepts mcps parameter with MCP server configs.
    
    This test replicates the exact code snippet from the documentation
    that was failing in issue #3858.
    """
    research_agent = Agent(
        role="Research Analyst",
        goal="Find and analyze information using advanced search tools",
        backstory="Expert researcher with access to multiple data sources",
        mcps=[
            MCPServerStdio(
                command="python",
                args=["local_server.py"],
                env={"API_KEY": "your_key"},
            ),
            MCPServerHTTP(
                url="https://api.research.com/mcp",
                headers={"Authorization": "Bearer your_token"},
            ),
        ]
    )
    
    assert research_agent.role == "Research Analyst"
    assert research_agent.goal == "Find and analyze information using advanced search tools"
    assert len(research_agent.mcps) == 2
    
    assert isinstance(research_agent.mcps[0], MCPServerStdio)
    assert isinstance(research_agent.mcps[1], MCPServerHTTP)


def test_documentation_example_full_workflow():
    """Test the complete workflow from the documentation that was failing in issue #3858.
    
    This test ensures that the exact code snippet from the documentation works correctly.
    Note: This test doesn't actually execute the crew (which would require real MCP servers),
    but verifies that the setup works as documented.
    """
    research_agent = Agent(
        role="Research Analyst",
        goal="Find and analyze information using advanced search tools",
        backstory="Expert researcher with access to multiple data sources",
        mcps=[
            MCPServerStdio(
                command="python",
                args=["local_server.py"],
                env={"API_KEY": "your_key"},
            ),
            MCPServerHTTP(
                url="https://api.research.com/mcp",
                headers={"Authorization": "Bearer your_token"},
            ),
        ]
    )

    research_task = Task(
        description="Research the latest developments in AI agent frameworks",
        expected_output="Comprehensive research report with citations",
        agent=research_agent
    )

    crew = Crew(agents=[research_agent], tasks=[research_task])
    
    assert crew is not None
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    assert crew.agents[0] == research_agent
    assert crew.tasks[0] == research_task
    


def test_mcp_server_configs_are_pydantic_models():
    """Test that MCP server configs are proper Pydantic models with validation."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        MCPServerStdio()  # Missing required 'command' field
    
    with pytest.raises(Exception):  # Pydantic ValidationError
        MCPServerHTTP()  # Missing required 'url' field
    
    with pytest.raises(Exception):  # Pydantic ValidationError
        MCPServerSSE()  # Missing required 'url' field


def test_mcp_server_stdio_with_optional_fields():
    """Test MCPServerStdio with optional fields."""
    mcp_server = MCPServerStdio(command="python")
    assert mcp_server.command == "python"
    assert mcp_server.args == []
    assert mcp_server.env is None
    
    mcp_server = MCPServerStdio(
        command="python",
        args=["server.py", "--port", "8000"],
        env={"API_KEY": "test", "DEBUG": "true"},
    )
    assert mcp_server.command == "python"
    assert mcp_server.args == ["server.py", "--port", "8000"]
    assert mcp_server.env == {"API_KEY": "test", "DEBUG": "true"}


def test_mcp_server_http_with_optional_fields():
    """Test MCPServerHTTP with optional fields."""
    mcp_server = MCPServerHTTP(url="https://api.example.com/mcp")
    assert mcp_server.url == "https://api.example.com/mcp"
    assert mcp_server.headers is None
    assert mcp_server.streamable is True  # Default value
    
    mcp_server = MCPServerHTTP(
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer token", "X-Custom": "value"},
        streamable=False,
    )
    assert mcp_server.url == "https://api.example.com/mcp"
    assert mcp_server.headers == {"Authorization": "Bearer token", "X-Custom": "value"}
    assert mcp_server.streamable is False


def test_mcp_server_sse_with_optional_fields():
    """Test MCPServerSSE with optional fields."""
    mcp_server = MCPServerSSE(url="https://api.example.com/mcp/sse")
    assert mcp_server.url == "https://api.example.com/mcp/sse"
    assert mcp_server.headers is None
    
    mcp_server = MCPServerSSE(
        url="https://api.example.com/mcp/sse",
        headers={"Authorization": "Bearer token"},
    )
    assert mcp_server.url == "https://api.example.com/mcp/sse"
    assert mcp_server.headers == {"Authorization": "Bearer token"}
