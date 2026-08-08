"""Tests for MCPToolResolver native (non-AMP) resolution paths."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerHTTP, MCPServerSSE
from crewai.mcp.tool_resolver import MCPToolResolver

# asyncio.run() in a worker thread needs localhost socketpair for the event loop.
pytestmark = pytest.mark.block_network(
    allowed_hosts=["127.0.0.1", "::1", "localhost"]
)


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


@pytest.fixture
def http_config():
    return MCPServerHTTP(url="https://mcp.example.com/api")


@pytest.fixture
def sse_config():
    return MCPServerSSE(url="https://mcp.example.com/sse")


def _mock_discovery_client(mock_client_class, tools):
    mock_client = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=tools)
    mock_client.connected = False
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client_class.return_value = mock_client
    return mock_client


def _close_coro_and_return(value):
    def _runner(coro):
        if asyncio.iscoroutine(coro):
            coro.close()
        return value

    return _runner


def _close_coro_and_raise(exc):
    def _runner(coro):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise exc

    return _runner


class TestResolveNativeEmptyTools:
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_logs_warning_and_returns_empty_when_server_has_no_tools(
        self, mock_client_class, mock_asyncio_run, resolver, http_config
    ):
        mock_client = AsyncMock()
        mock_client.connected = False
        mock_client_class.return_value = mock_client
        mock_asyncio_run.side_effect = _close_coro_and_return([])

        mock_log = MagicMock()
        resolver._logger = MagicMock(log=mock_log)

        tools, clients = resolver._resolve_native(http_config)

        assert tools == []
        assert clients == []
        warning_calls = [
            call for call in mock_log.call_args_list if call.args[0] == "warning"
        ]
        assert any(
            "No tools discovered from MCP server" in call.args[1]
            for call in warning_calls
        )

    @patch("crewai.mcp.tool_resolver.asyncio.run")
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_logs_warning_when_tool_filter_removes_all_tools(
        self, mock_client_class, mock_asyncio_run, resolver
    ):
        mock_client = AsyncMock()
        mock_client.connected = False
        mock_client_class.return_value = mock_client
        mock_asyncio_run.side_effect = _close_coro_and_return(
            [{"name": "search", "description": "Search"}]
        )

        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=lambda _tool: False,
        )

        mock_log = MagicMock()
        resolver._logger = MagicMock(log=mock_log)

        tools, clients = resolver._resolve_native(config)

        assert tools == []
        assert clients == []
        warning_calls = [
            call for call in mock_log.call_args_list if call.args[0] == "warning"
        ]
        assert any(
            "No tools discovered from MCP server" in call.args[1]
            for call in warning_calls
        )


class TestResolveNativeRuntimeError:
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    def test_unmatched_runtime_error_is_wrapped_not_swallowed(
        self, mock_asyncio_run, resolver, http_config
    ):
        mock_asyncio_run.side_effect = _close_coro_and_raise(
            RuntimeError("some other failure")
        )

        with pytest.raises(RuntimeError, match="Failed to get native MCP tools"):
            resolver._resolve_native(http_config)

    @pytest.mark.parametrize(
        "config_fixture",
        ["http_config", "sse_config"],
    )
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_resolves_tools_from_running_event_loop(
        self, mock_client_class, resolver, config_fixture, request
    ):
        """Flow runtime already owns an event loop; discovery must complete via
        a real worker thread without calling asyncio.run on the caller (#6843)."""
        config = request.getfixturevalue(config_fixture)
        mock_client = _mock_discovery_client(
            mock_client_class,
            [
                {
                    "name": "search",
                    "description": "Search",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ],
        )

        async def _call_from_running_loop():
            return resolver._resolve_native(config)

        tools, clients = asyncio.run(_call_from_running_loop())

        assert clients == []
        assert len(tools) == 1
        assert "search" in tools[0].name
        mock_client.connect.assert_awaited()
        mock_client.list_tools.assert_awaited()
        mock_client.disconnect.assert_awaited()

    @patch("crewai.mcp.tool_resolver.MCPToolResolver._run_coro_sync")
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_cleanup_uses_loop_safe_runner_when_discovery_fails(
        self, mock_client_class, mock_run_coro_sync, resolver, http_config
    ):
        mock_client = AsyncMock()
        mock_client.connected = True
        mock_client_class.return_value = mock_client

        mock_run_coro_sync.side_effect = [
            RuntimeError("discovery blew up"),
            None,
        ]

        with pytest.raises(RuntimeError, match="Failed to get native MCP tools"):
            resolver._resolve_native(http_config)

        assert mock_run_coro_sync.call_count == 2
        cleanup_coro = mock_run_coro_sync.call_args_list[1].args[0]
        assert asyncio.iscoroutine(cleanup_coro)
        cleanup_coro.close()
        first_coro = mock_run_coro_sync.call_args_list[0].args[0]
        if asyncio.iscoroutine(first_coro):
            first_coro.close()