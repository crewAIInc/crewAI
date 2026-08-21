"""Regression tests for GitHub issue #6843.

Native MCP HTTP / streamable discovery is invoked from sync code while a
CrewAI Flow already has an event loop running. ``asyncio.run()`` must not
be used on that thread, and a connection failure must not be rewritten
into "cannot be called from a running event loop".
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crewai.mcp.config import MCPServerHTTP
from crewai.mcp.tool_resolver import MCPToolResolver


def _resolver() -> MCPToolResolver:
    return MCPToolResolver(agent=MagicMock(), logger=MagicMock())


def _http_config() -> MCPServerHTTP:
    return MCPServerHTTP(
        url="https://mcp.example.com/mcp",
        streamable=True,
        cache_tools_list=True,
    )


def _mock_client(
    *,
    tools: list | None = None,
    connect_error: Exception | None = None,
) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.connected = False
    if connect_error is not None:
        mock_client.connect = AsyncMock(side_effect=connect_error)
    else:
        mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(return_value=tools or [])
    mock_client.disconnect = AsyncMock()
    return mock_client


class TestResolveNativeFromRunningLoop:
    @pytest.mark.asyncio
    @patch("crewai.mcp.tool_resolver.MCPClient")
    async def test_does_not_call_asyncio_run_when_loop_already_running(
        self, mock_client_class: MagicMock
    ) -> None:
        mock_client_class.return_value = _mock_client()
        resolver = _resolver()
        assert asyncio.get_running_loop() is not None

        with patch(
            "crewai.mcp.tool_resolver.asyncio.run", wraps=asyncio.run
        ) as mock_run:
            tools = resolver.resolve([_http_config()])

        assert tools == []
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    @patch("crewai.mcp.tool_resolver.MCPClient")
    async def test_connection_error_does_not_raise_nested_event_loop_error(
        self, mock_client_class: MagicMock
    ) -> None:
        mock_client_class.return_value = _mock_client(
            connect_error=ConnectionError("Session terminated")
        )
        resolver = _resolver()
        assert asyncio.get_running_loop() is not None

        with pytest.raises(
            RuntimeError, match="Failed to get native MCP tools"
        ) as exc_info:
            resolver._resolve_native(_http_config())

        messages = [str(exc_info.value)]
        cause = exc_info.value.__cause__
        while cause is not None:
            messages.append(str(cause))
            cause = cause.__cause__
        blob = "\n".join(messages)
        assert "Session terminated" in blob
        assert "cannot be called from a running event loop" not in blob
