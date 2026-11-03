"""Tests for async tool_filter handling in MCPToolResolver."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerHTTP
from crewai.mcp.tool_resolver import MCPToolResolver


@pytest.fixture
def agent():
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )


@pytest.fixture
def resolver(agent):
    return MCPToolResolver(agent=agent, logger=agent._logger)


def _mock_client(mock_client_class, tool_names):
    mock_client = AsyncMock()
    mock_client.list_tools = AsyncMock(
        return_value=[
            {"name": name, "description": f"{name} tool"} for name in tool_names
        ]
    )
    mock_client.connected = False
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client_class.return_value = mock_client


class TestAsyncToolFilter:
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_async_filter_blocks_tools_its_meant_to_block(
        self, mock_client_class, resolver
    ):
        """An async tool_filter must have its result awaited.

        An unawaited coroutine is always truthy, so every tool passed the
        filter before the fix (issue: "Async MCP tool_filter is never
        awaited, so every tool passes the filter").
        """
        _mock_client(
            mock_client_class, tool_names=["read_file", "danger_delete"]
        )

        async def block_dangerous(_context, tool):
            return not tool["name"].startswith("danger_")

        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=block_dangerous,
        )

        tools, _clients = resolver._resolve_native(config)

        assert [tool.name for tool in tools] == ["mcp_example_com_api_read_file"]

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_async_single_argument_filter_blocks_tools(
        self, mock_client_class, resolver
    ):
        _mock_client(
            mock_client_class, tool_names=["read_file", "danger_delete"]
        )

        async def block_dangerous(tool):
            return not tool["name"].startswith("danger_")

        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=block_dangerous,
        )

        tools, _clients = resolver._resolve_native(config)

        assert [tool.name for tool in tools] == ["mcp_example_com_api_read_file"]

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_sync_filter_still_works(self, mock_client_class, resolver):
        _mock_client(
            mock_client_class, tool_names=["read_file", "danger_delete"]
        )

        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=lambda tool: not tool["name"].startswith("danger_"),
        )

        tools, _clients = resolver._resolve_native(config)

        assert [tool.name for tool in tools] == ["mcp_example_com_api_read_file"]
