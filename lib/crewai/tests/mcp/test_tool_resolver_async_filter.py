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

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_type_error_from_filter_body_is_not_mistaken_for_arity(
        self, mock_client_class, resolver
    ):
        """A two-argument filter that raises TypeError must not be called again.

        Probing the arity by retrying on TypeError would call the filter a second
        time with one argument: the real error would be replaced by a signature
        error and the filter's side effects would run twice.
        """
        _mock_client(mock_client_class, tool_names=["read_file"])
        calls = []

        def broken_filter(_context, tool):
            calls.append(tool["name"])
            raise TypeError("boom from inside the filter")

        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=broken_filter,
        )

        with pytest.raises(ConnectionError) as excinfo:
            resolver._resolve_native(config)

        # The failure that surfaces must be the filter's own TypeError, not the
        # "missing 1 required positional argument" one a blind retry would raise.
        assert isinstance(excinfo.value.__cause__, TypeError)
        assert str(excinfo.value.__cause__) == "boom from inside the filter"
        assert calls == ["read_file"]

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_filter_returning_a_nested_awaitable_is_resolved(
        self, mock_client_class, resolver
    ):
        """A coroutine that resolves to another coroutine must be awaited twice.

        Stopping one level in would call bool() on the inner coroutine, which is
        truthy no matter what verdict the filter meant to return.
        """
        _mock_client(
            mock_client_class, tool_names=["read_file", "danger_delete"]
        )

        async def verdict(_tool):
            return False

        def nested_filter(_context, tool):
            async def outer():
                return verdict(tool)

            return outer()

        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=nested_filter,
        )

        tools, _clients = resolver._resolve_native(config)

        assert tools == []
