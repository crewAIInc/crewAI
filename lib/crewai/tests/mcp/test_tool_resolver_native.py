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
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_logs_warning_and_returns_empty_when_server_has_no_tools(
        self, mock_client_class, resolver, http_config
    ):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

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

    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_logs_warning_when_tool_filter_removes_all_tools(
        self, mock_client_class, resolver
    ):
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "search", "description": "Search"}]
        )
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

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

    @patch("concurrent.futures.ThreadPoolExecutor")
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    @patch.object(asyncio, "get_running_loop", return_value=MagicMock())
    def test_thread_pool_runtimeerror_not_mistaken_for_no_event_loop(
        self, mock_get_loop, mock_asyncio_run, mock_executor, resolver, http_config
    ):
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError(
            "Error during setup client and list tools: connection refused"
        )

        def _submit_side_effect(fn, *args, **kwargs):
            coro = args[1] if len(args) > 1 else kwargs.get("coro")
            if coro is not None:
                coro.close()
            return mock_future

        mock_executor.return_value.__enter__.return_value.submit.side_effect = (
            _submit_side_effect
        )

        with pytest.raises(RuntimeError, match="Failed to get native MCP tools"):
            resolver._resolve_native(http_config)

        mock_asyncio_run.assert_not_called()

    @patch("concurrent.futures.ThreadPoolExecutor")
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    @patch.object(asyncio, "get_running_loop", return_value=MagicMock())
    def test_thread_pool_cancellederror_handled(
        self, mock_get_loop, mock_asyncio_run, mock_executor, resolver, http_config
    ):
        mock_future = MagicMock()
        mock_future.result.side_effect = asyncio.CancelledError("task cancelled")

        def _submit_side_effect(fn, *args, **kwargs):
            coro = args[1] if len(args) > 1 else kwargs.get("coro")
            if coro is not None:
                coro.close()
            return mock_future

        mock_executor.return_value.__enter__.return_value.submit.side_effect = (
            _submit_side_effect
        )

        with pytest.raises(ConnectionError, match="MCP connection was cancelled"):
            resolver._resolve_native(http_config)

        mock_asyncio_run.assert_not_called()