import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, patch

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


def _make_mock_client(tool_definitions):
    """Create a mock MCPClient that returns *tool_definitions*."""
    client = AsyncMock()
    client.list_tools = AsyncMock(return_value=tool_definitions)
    client.connected = False
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.call_tool = AsyncMock(return_value="test result")
    return client


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

    with patch("crewai.mcp.tool_resolver.MCPClient") as mock_client_class:
        mock_client_class.return_value = _make_mock_client(mock_tool_definitions)

        tools = agent.get_mcp_tools([stdio_config])

        assert len(tools) == 2
        assert all(isinstance(tool, BaseTool) for tool in tools)

        mock_client_class.assert_called_once()
        transport = mock_client_class.call_args.kwargs["transport"]
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

    with patch("crewai.mcp.tool_resolver.MCPClient") as mock_client_class:
        mock_client_class.return_value = _make_mock_client(mock_tool_definitions)

        tools = agent.get_mcp_tools([http_config])

        assert len(tools) == 2
        assert all(isinstance(tool, BaseTool) for tool in tools)

        mock_client_class.assert_called_once()
        transport = mock_client_class.call_args.kwargs["transport"]
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

    with patch("crewai.mcp.tool_resolver.MCPClient") as mock_client_class:
        mock_client_class.return_value = _make_mock_client(mock_tool_definitions)

        tools = agent.get_mcp_tools([sse_config])

        assert len(tools) == 2
        assert all(isinstance(tool, BaseTool) for tool in tools)

        mock_client_class.assert_called_once()
        transport = mock_client_class.call_args.kwargs["transport"]
        assert transport.url == "https://api.example.com/mcp/sse"
        assert transport.headers == {"Authorization": "Bearer test_token"}


def test_mcp_tool_execution_in_sync_context(mock_tool_definitions):
    """Test MCPNativeTool execution in synchronous context (normal crew execution)."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    with patch("crewai.mcp.tool_resolver.MCPClient") as mock_client_class:
        mock_client_class.return_value = _make_mock_client(mock_tool_definitions)

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
        # 1 discovery + 1 for the run() invocation
        assert mock_client_class.call_count == 2


@pytest.mark.asyncio
async def test_mcp_tool_execution_in_async_context(mock_tool_definitions):
    """Test MCPNativeTool execution in async context (e.g., from a Flow)."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    with patch("crewai.mcp.tool_resolver.MCPClient") as mock_client_class:
        mock_client_class.return_value = _make_mock_client(mock_tool_definitions)

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
        assert mock_client_class.call_count == 2


def test_each_invocation_gets_fresh_client(mock_tool_definitions):
    """Every tool.run() must create its own MCPClient (no shared state)."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    clients_created: list = []

    def _make_client(**kwargs):
        client = _make_mock_client(mock_tool_definitions)
        clients_created.append(client)
        return client

    with patch("crewai.mcp.tool_resolver.MCPClient", side_effect=_make_client):
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[http_config],
        )

        tools = agent.get_mcp_tools([http_config])
        assert len(tools) == 2
        # 1 discovery client so far
        assert len(clients_created) == 1

        # Two sequential calls to the same tool must create 2 new clients
        tools[0].run(query="q1")
        tools[0].run(query="q2")
        assert len(clients_created) == 3
        assert clients_created[1] is not clients_created[2]


def test_parallel_mcp_tool_execution_same_tool(mock_tool_definitions):
    """Parallel calls to the *same* tool must not interfere."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    call_log: list[str] = []

    def _make_client(**kwargs):
        client = AsyncMock()
        client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        client.connected = False
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()

        async def _call_tool(name, args):
            call_log.append(name)
            await asyncio.sleep(0.05)
            return f"result-{name}"

        client.call_tool = AsyncMock(side_effect=_call_tool)
        return client

    with patch("crewai.mcp.tool_resolver.MCPClient", side_effect=_make_client):
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[http_config],
        )

        tools = agent.get_mcp_tools([http_config])
        assert len(tools) >= 1
        tool = tools[0]

        # Call the SAME tool concurrently -- the exact scenario from the bug
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(tool.run, query="q1"),
                pool.submit(tool.run, query="q2"),
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 2
    assert all("result-" in r for r in results)
    assert len(call_log) == 2


def test_parallel_mcp_tool_execution_different_tools(mock_tool_definitions):
    """Parallel calls to different tools from the same server must not interfere."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    call_log: list[str] = []

    def _make_client(**kwargs):
        client = AsyncMock()
        client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        client.connected = False
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()

        async def _call_tool(name, args):
            call_log.append(name)
            await asyncio.sleep(0.05)
            return f"result-{name}"

        client.call_tool = AsyncMock(side_effect=_call_tool)
        return client

    with patch("crewai.mcp.tool_resolver.MCPClient", side_effect=_make_client):
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[http_config],
        )

        tools = agent.get_mcp_tools([http_config])
        assert len(tools) == 2

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(tools[0].run, query="q1"),
                pool.submit(tools[1].run, query="q2"),
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 2
    assert all("result-" in r for r in results)
    assert len(call_log) == 2
