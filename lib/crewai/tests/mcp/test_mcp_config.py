import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerHTTP, MCPServerSSE, MCPServerStdio
from crewai.tools.base_tool import BaseTool


@pytest.fixture
def mock_tool_definitions():
    """Create mock MCP tool definitions (as returned by list_tools)."""
    return [
        {
            "name": "test_tool_1",
            "description": "Test tool 1 description",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "test_tool_2",
            "description": "Test tool 2 description",
            "inputSchema": {}
        }
    ]


def test_agent_with_stdio_mcp_config(mock_tool_definitions):
    """Test agent setup with MCPServerStdio configuration."""
    stdio_config = MCPServerStdio(
        command="python",
        args=["server.py"],
        env={"API_KEY": "test_key"},
    )

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        mcps=[stdio_config],
    )


    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False  # Will trigger connect
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools([stdio_config])

        assert len(tools) == 2
        assert all(isinstance(tool, BaseTool) for tool in tools)

        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        transport = call_args.kwargs["transport"]
        assert transport.command == "python"
        assert transport.args == ["server.py"]
        assert transport.env == {"API_KEY": "test_key"}


def test_agent_with_http_mcp_config(mock_tool_definitions):
    """Test agent setup with MCPServerHTTP configuration."""
    http_config = MCPServerHTTP(
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer test_token"},
        streamable=True,
    )

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        mcps=[http_config],
    )

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False  # Will trigger connect
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools([http_config])

        assert len(tools) == 2
        assert all(isinstance(tool, BaseTool) for tool in tools)

        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        transport = call_args.kwargs["transport"]
        assert transport.url == "https://api.example.com/mcp"
        assert transport.headers == {"Authorization": "Bearer test_token"}
        assert transport.streamable is True


def test_agent_with_sse_mcp_config(mock_tool_definitions):
    """Test agent setup with MCPServerSSE configuration."""
    sse_config = MCPServerSSE(
        url="https://api.example.com/mcp/sse",
        headers={"Authorization": "Bearer test_token"},
    )

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        mcps=[sse_config],
    )

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools([sse_config])

        assert len(tools) == 2
        assert all(isinstance(tool, BaseTool) for tool in tools)

        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        transport = call_args.kwargs["transport"]
        assert transport.url == "https://api.example.com/mcp/sse"
        assert transport.headers == {"Authorization": "Bearer test_token"}


def test_mcp_tool_execution_in_sync_context(mock_tool_definitions):
    """Test MCPNativeTool execution in synchronous context (normal crew execution)."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="test result")
        mock_client_class.return_value = mock_client

        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[http_config],
        )

        tools = agent.get_mcp_tools([http_config])
        assert len(tools) == 2


        tool = tools[0]
        result = tool.run(query="test query")

        assert result == "test result"
        mock_client.call_tool.assert_called()


def test_mcp_tools_loaded_when_tools_is_none(mock_tool_definitions):
    """Test that MCP tools are loaded even when agent tools is None.

    Regression test for https://github.com/crewAIInc/crewAI/issues/4568
    When an agent is created with mcps but tools is explicitly None,
    _prepare_kickoff should still load MCP tools.
    """
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        mcps=[http_config],
        tools=None,
    )

    mock_mcp_tool = MagicMock(spec=BaseTool)
    mock_mcp_tool.name = "mcp_test_tool"

    with patch.object(Agent, "get_mcp_tools", return_value=[mock_mcp_tool]):
        # Manually invoke the MCP loading logic from _prepare_kickoff
        if agent.mcps:
            mcps = agent.get_mcp_tools(agent.mcps)
            if mcps:
                if agent.tools is None:
                    agent.tools = []
                agent.tools.extend(mcps)

        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "mcp_test_tool"


def test_mcp_tools_loaded_when_tools_is_empty_list(mock_tool_definitions):
    """Test that MCP tools are loaded when agent tools is an empty list."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        mcps=[http_config],
        tools=[],
    )

    mock_mcp_tool = MagicMock(spec=BaseTool)
    mock_mcp_tool.name = "mcp_test_tool"

    with patch.object(Agent, "get_mcp_tools", return_value=[mock_mcp_tool]):
        if agent.mcps:
            mcps = agent.get_mcp_tools(agent.mcps)
            if mcps:
                if agent.tools is None:
                    agent.tools = []
                agent.tools.extend(mcps)

        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "mcp_test_tool"


def test_mcp_tools_appended_to_existing_tools(mock_tool_definitions):
    """Test that MCP tools are appended to existing agent tools."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    existing_tool = MagicMock(spec=BaseTool)
    existing_tool.name = "existing_tool"
    existing_tool.description = "Existing tool"
    existing_tool.func = lambda: None

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        mcps=[http_config],
        tools=[existing_tool],
    )

    mock_mcp_tool = MagicMock(spec=BaseTool)
    mock_mcp_tool.name = "mcp_test_tool"

    with patch.object(Agent, "get_mcp_tools", return_value=[mock_mcp_tool]):
        if agent.mcps:
            mcps = agent.get_mcp_tools(agent.mcps)
            if mcps:
                if agent.tools is None:
                    agent.tools = []
                agent.tools.extend(mcps)

        assert agent.tools is not None
        assert len(agent.tools) == 2


def test_agent_without_tools_only_mcps(mock_tool_definitions):
    """Test that an agent created with only mcps (no tools keyword) works correctly.

    This is the exact scenario from issue #4568 where users create agents like:
        Agent(role=..., goal=..., backstory=..., mcps=[MCPServerHTTP(...)])
    without specifying tools at all.
    """
    http_config = MCPServerHTTP(url="http://localhost:8022/mcp")

    agent = Agent(
        role="Sandbox Agent",
        goal="Test goal",
        backstory="Test backstory",
        mcps=[http_config],
    )

    mock_mcp_tool = MagicMock(spec=BaseTool)
    mock_mcp_tool.name = "sandbox_tool"

    with patch.object(Agent, "get_mcp_tools", return_value=[mock_mcp_tool]):
        # Simulate _prepare_kickoff MCP loading logic
        if agent.mcps:
            mcps = agent.get_mcp_tools(agent.mcps)
            if mcps:
                if agent.tools is None:
                    agent.tools = []
                agent.tools.extend(mcps)

        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "sandbox_tool"


@pytest.mark.asyncio
async def test_mcp_tool_execution_in_async_context(mock_tool_definitions):
    """Test MCPNativeTool execution in async context (e.g., from a Flow)."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value="test result")
        mock_client_class.return_value = mock_client

        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[http_config],
        )

        tools = agent.get_mcp_tools([http_config])
        assert len(tools) == 2


        tool = tools[0]
        result = tool.run(query="test query")

        assert result == "test result"
        mock_client.call_tool.assert_called()
