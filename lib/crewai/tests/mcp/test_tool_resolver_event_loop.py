"""Tests for MCPToolResolver behavior inside an active asyncio event loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.mcp.config import MCPServerHTTP
from crewai.mcp.exceptions import MCPConnectionError
from crewai.mcp.tool_resolver import MCPToolResolver, _mcp_schema_cache


@pytest.fixture
def resolver():
    from crewai.agent.core import Agent

    agent = Agent(role="Test Agent", goal="Test goal", backstory="Test backstory")
    return MCPToolResolver(agent=agent, logger=agent._logger)


class TestMCPToolResolverRunningEventLoop:
    @pytest.mark.asyncio
    async def test_cleanup_disconnects_all_clients_when_loop_is_running(self, resolver):
        failing_client = MagicMock()
        failing_client.connected = True
        failing_client.disconnect = AsyncMock(side_effect=RuntimeError("boom"))

        healthy_client = MagicMock()
        healthy_client.connected = True
        healthy_client.disconnect = AsyncMock()

        resolver._clients.extend([failing_client, healthy_client])

        resolver.cleanup()

        failing_client.disconnect.assert_awaited_once()
        healthy_client.disconnect.assert_awaited_once()
        assert resolver.clients == []

    @pytest.mark.asyncio
    async def test_get_mcp_tool_schemas_succeeds_when_loop_is_running(self, resolver):
        server_url = "https://mcp-running-loop.example.com/api"
        _mcp_schema_cache.pop(server_url, None)
        expected = {"search": {"description": "Search docs", "args_schema": None}}

        try:
            with patch.object(
                resolver,
                "_get_mcp_tool_schemas_async",
                new=AsyncMock(return_value=expected),
            ) as mock_discover:
                schemas = resolver._get_mcp_tool_schemas({"url": server_url})

            assert schemas == expected
            mock_discover.assert_awaited_once_with({"url": server_url})
        finally:
            _mcp_schema_cache.pop(server_url, None)

    @pytest.mark.asyncio
    async def test_resolve_native_disconnects_on_failure_when_loop_is_running(
        self, resolver
    ):
        mock_client = AsyncMock()
        mock_client.connected = False

        async def _connect():
            mock_client.connected = True

        mock_client.connect = AsyncMock(side_effect=_connect)
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "ping", "description": "Ping"}]
        )

        disconnect_calls = 0

        async def _disconnect():
            nonlocal disconnect_calls
            disconnect_calls += 1
            if disconnect_calls == 1:
                # Simulate connect/list_tools cleanup leaving client connected
                raise RuntimeError("initial disconnect failed")
            mock_client.connected = False

        def _failing_filter(_tool):
            raise ValueError("bad filter")

        mock_client.disconnect = AsyncMock(side_effect=_disconnect)
        config = MCPServerHTTP(
            url="https://mcp.example.com/api",
            tool_filter=_failing_filter,
        )

        with (
            patch("crewai.mcp.tool_resolver.MCPClient", return_value=mock_client),
            pytest.raises(
                (ConnectionError, MCPConnectionError),
                match="Failed to get native MCP tools",
            ),
        ):
            resolver._resolve_native(config)

        assert disconnect_calls == 2
        assert mock_client.connected is False
