"""Tests for MCPToolResolver native (non-AMP) resolution paths."""

import asyncio
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


@pytest.fixture
def http_config():
    return MCPServerHTTP(url="https://mcp.example.com/api")


class TestResolveNativeEmptyTools:
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_logs_warning_and_returns_empty_when_server_has_no_tools(
        self, mock_client_class, mock_asyncio_run, resolver, http_config
    ):
        mock_client = AsyncMock()
        mock_client.connected = False
        mock_client_class.return_value = mock_client
        mock_asyncio_run.return_value = []

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
        mock_asyncio_run.return_value = [{"name": "search", "description": "Search"}]

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
        mock_asyncio_run.side_effect = RuntimeError("some other failure")

        with pytest.raises(RuntimeError, match="Failed to get native MCP tools"):
            resolver._resolve_native(http_config)

    @patch("crewai.mcp.tool_resolver.MCPClient")
    @patch("crewai.mcp.tool_resolver.concurrent.futures.ThreadPoolExecutor")
    @patch("crewai.mcp.tool_resolver.asyncio.get_running_loop")
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    def test_uses_worker_thread_when_event_loop_already_running(
        self,
        mock_asyncio_run,
        mock_get_running_loop,
        mock_executor_cls,
        mock_client_class,
        resolver,
        http_config,
    ):
        """Flow runtime already owns an event loop; discovery must run
        asyncio.run in a worker thread, not on the caller's loop (#6843)."""
        mock_get_running_loop.return_value = MagicMock(is_running=MagicMock(return_value=True))
        mock_asyncio_run.return_value = [
            {
                "name": "search",
                "description": "Search",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

        mock_future = MagicMock()
        mock_future.result.return_value = mock_asyncio_run.return_value
        mock_executor = MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = False
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        mock_client = AsyncMock()
        mock_client.connected = False
        mock_client_class.return_value = mock_client

        tools, clients = resolver._resolve_native(http_config)

        assert clients == []
        assert len(tools) == 1
        mock_executor.submit.assert_called_once()
        # asyncio.run must not be invoked on this thread when a loop is running
        mock_asyncio_run.assert_not_called()