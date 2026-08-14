"""Tests for MCPToolResolver native (non-AMP) resolution paths."""

import threading
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


class _ThreadAffinityFakeClient:
    """Fake MCP client that raises when awaited from the main thread.

    Simulates the thread-affinity constraint of real asyncio objects:
    async methods must run in a worker thread, not on the main thread
    where an event loop is already running.
    """

    def __init__(self, *args, **kwargs):
        self._main_tid = threading.current_thread().ident
        self.connected = False
        self._executor_tid = None

    def _check_affinity(self):
        if (
            threading.current_thread().ident == self._main_tid
            and _loop_is_running()
        ):
            raise RuntimeError(
                "Event loop affinity: async operations must run in a worker thread"
            )

    async def connect(self):
        self._check_affinity()
        self._executor_tid = threading.current_thread().ident
        self.connected = True

    async def disconnect(self):
        self._check_affinity()
        self.connected = False

    async def list_tools(self):
        self._check_affinity()
        return [{"name": "test_tool", "description": "A test tool", "inputSchema": {}}]


def _loop_is_running():
    try:
        import asyncio
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class TestResolveNativeAsyncioThreadAffinity:
    @patch("crewai.mcp.tool_resolver.MCPClient", _ThreadAffinityFakeClient)
    def test_resolve_native_from_running_loop_uses_executor_path(
        self, resolver, http_config
    ):
        """Test that _resolve_native works when called from a running event loop.

        Uses a thread-aware fake that raises a thread-affinity error unless
        async methods execute in a worker thread (i.e. via the executor path).
        """
        import asyncio

        mock_log = MagicMock()
        resolver._logger = MagicMock(log=mock_log)

        async def _call_resolve_native():
            return resolver._resolve_native(http_config)

        tools, clients = asyncio.run(_call_resolve_native())

        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert clients == []