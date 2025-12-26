import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.agent.utils import prepare_tools
from crewai.mcp.config import MCPServerHTTP, MCPServerSSE, MCPServerStdio
from crewai.task import Task
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


def test_prepare_tools_injects_mcp_tools(mock_tool_definitions):
    """Test that prepare_tools injects MCP tools when agent has mcps configured.

    This is the core fix for issue #4133 - LLM doesn't see MCP tools when
    using standalone agent execution (without Crew).
    """
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[http_config],
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        final_tools = prepare_tools(agent, None, task)

        assert len(final_tools) == 2
        assert all(isinstance(tool, BaseTool) for tool in final_tools)
        tool_names = [tool.name for tool in final_tools]
        assert any("test_tool_1" in name for name in tool_names)
        assert any("test_tool_2" in name for name in tool_names)


def test_prepare_tools_merges_mcp_tools_with_existing_tools(mock_tool_definitions):
    """Test that prepare_tools merges MCP tools with existing agent tools.

    MCP tools are added alongside existing tools. Note that MCP tools have
    prefixed names (based on server URL), so they won't conflict with
    existing tools that have the same base name.
    """
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    class ExistingTool(BaseTool):
        name: str = "existing_tool"
        description: str = "An existing tool"

        def _run(self, **kwargs):
            return "existing result"

    class AnotherTool(BaseTool):
        name: str = "another_tool"
        description: str = "Another existing tool"

        def _run(self, **kwargs):
            return "another result"

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            tools=[ExistingTool(), AnotherTool()],
            mcps=[http_config],
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        final_tools = prepare_tools(agent, None, task)

        assert len(final_tools) == 4
        tool_names = [tool.name for tool in final_tools]
        assert "existing_tool" in tool_names
        assert "another_tool" in tool_names
        assert any("test_tool_1" in name for name in tool_names)
        assert any("test_tool_2" in name for name in tool_names)


def test_prepare_tools_does_not_mutate_original_tools_list(mock_tool_definitions):
    """Test that prepare_tools does not mutate the original tools list."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    class ExistingTool(BaseTool):
        name: str = "existing_tool"
        description: str = "An existing tool"

        def _run(self, **kwargs):
            return "existing result"

    original_tools = [ExistingTool()]
    original_tools_copy = list(original_tools)

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            tools=original_tools,
            mcps=[http_config],
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        final_tools = prepare_tools(agent, original_tools, task)

        assert len(original_tools) == len(original_tools_copy)
        assert len(final_tools) == 3


def test_prepare_tools_handles_mcp_failure_gracefully(mock_tool_definitions):
    """Test that prepare_tools continues without MCP tools if get_mcp_tools fails."""
    http_config = MCPServerHTTP(url="https://api.example.com/mcp")

    class ExistingTool(BaseTool):
        name: str = "existing_tool"
        description: str = "An existing tool"

        def _run(self, **kwargs):
            return "existing result"

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client_class.side_effect = Exception("Connection failed")

        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            tools=[ExistingTool()],
            mcps=[http_config],
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        final_tools = prepare_tools(agent, None, task)

        assert len(final_tools) == 1
        assert final_tools[0].name == "existing_tool"


def test_prepare_tools_without_mcps():
    """Test that prepare_tools works normally when agent has no mcps configured."""
    class ExistingTool(BaseTool):
        name: str = "existing_tool"
        description: str = "An existing tool"

        def _run(self, **kwargs):
            return "existing result"

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        tools=[ExistingTool()],
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    final_tools = prepare_tools(agent, None, task)

    assert len(final_tools) == 1
    assert final_tools[0].name == "existing_tool"
