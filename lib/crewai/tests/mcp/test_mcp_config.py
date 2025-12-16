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


def test_http_mcp_config_verify_default():
    """Test MCPServerHTTP has verify=True by default."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")
    assert http_config.verify is True


def test_http_mcp_config_verify_false():
    """Test MCPServerHTTP with verify=False."""
    http_config = MCPServerHTTP(
        url="https://api.example.com/mcp",
        verify=False,
    )
    assert http_config.verify is False


def test_http_mcp_config_verify_ca_bundle():
    """Test MCPServerHTTP with custom CA bundle path."""
    http_config = MCPServerHTTP(
        url="https://api.example.com/mcp",
        verify="/path/to/ca-bundle.crt",
    )
    assert http_config.verify == "/path/to/ca-bundle.crt"


def test_sse_mcp_config_verify_default():
    """Test MCPServerSSE has verify=True by default."""
    sse_config = MCPServerSSE(url="https://api.example.com/mcp/sse")
    assert sse_config.verify is True


def test_sse_mcp_config_verify_false():
    """Test MCPServerSSE with verify=False."""
    sse_config = MCPServerSSE(
        url="https://api.example.com/mcp/sse",
        verify=False,
    )
    assert sse_config.verify is False


def test_sse_mcp_config_verify_ca_bundle():
    """Test MCPServerSSE with custom CA bundle path."""
    sse_config = MCPServerSSE(
        url="https://api.example.com/mcp/sse",
        verify="/path/to/ca-bundle.crt",
    )
    assert sse_config.verify == "/path/to/ca-bundle.crt"


def test_agent_with_http_mcp_config_verify_false(mock_tool_definitions):
    """Test agent setup with MCPServerHTTP configuration with verify=False."""
    http_config = MCPServerHTTP(
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer test_token"},
        verify=False,
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
        mock_client.connected = False
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
        assert transport.verify is False


def test_agent_with_http_mcp_config_verify_ca_bundle(mock_tool_definitions):
    """Test agent setup with MCPServerHTTP configuration with custom CA bundle."""
    http_config = MCPServerHTTP(
        url="https://api.example.com/mcp",
        verify="/path/to/ca-bundle.crt",
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
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        tools = agent.get_mcp_tools([http_config])

        assert len(tools) == 2

        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        transport = call_args.kwargs["transport"]
        assert transport.verify == "/path/to/ca-bundle.crt"


def test_agent_with_sse_mcp_config_verify_false(mock_tool_definitions):
    """Test agent setup with MCPServerSSE configuration with verify=False."""
    sse_config = MCPServerSSE(
        url="https://api.example.com/mcp/sse",
        headers={"Authorization": "Bearer test_token"},
        verify=False,
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
        assert transport.verify is False


def test_agent_with_sse_mcp_config_verify_ca_bundle(mock_tool_definitions):
    """Test agent setup with MCPServerSSE configuration with custom CA bundle."""
    sse_config = MCPServerSSE(
        url="https://api.example.com/mcp/sse",
        verify="/path/to/ca-bundle.crt",
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

        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        transport = call_args.kwargs["transport"]
        assert transport.verify == "/path/to/ca-bundle.crt"
