"""Tests for MCPToolResolver authentication error handling."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from crewai.mcp.config import MCPServerHTTP
from crewai.mcp.exceptions import MCPAuthenticationError
from crewai.mcp.tool_resolver import MCPToolResolver


@pytest.fixture
def resolver():
    from crewai.agent.core import Agent

    agent = Agent(role="Test Agent", goal="Test goal", backstory="Test backstory")
    return MCPToolResolver(agent=agent, logger=agent._logger)


@pytest.fixture
def http_config():
    return MCPServerHTTP(url="https://mcp.example.com/api")


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mcp.example.com/mcp")
    response = httpx.Response(status_code, text="refused", request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


class TestResolveNativeAuthErrors:
    @patch("crewai.mcp.tool_resolver.asyncio.run")
    def test_cancelled_error_with_auth_status_raises_authentication_error(
        self, mock_asyncio_run, resolver, http_config
    ):
        cancelled = asyncio.CancelledError()
        cancelled.__context__ = _http_status_error(401)
        mock_asyncio_run.side_effect = cancelled

        with pytest.raises(MCPAuthenticationError) as exc_info:
            resolver._resolve_native(http_config)

        assert exc_info.value.status_code == 401
        assert "401 Unauthorized" in str(exc_info.value)
        assert "may indicate an authentication error" not in str(exc_info.value).lower()

    @patch("crewai.mcp.tool_resolver.asyncio.run")
    def test_typed_authentication_error_propagates_without_speculative_wording(
        self, mock_asyncio_run, resolver, http_config
    ):
        mock_asyncio_run.side_effect = MCPAuthenticationError(401)

        with pytest.raises(MCPAuthenticationError) as exc_info:
            resolver._resolve_native(http_config)

        assert exc_info.value.status_code == 401
        assert "may indicate" not in str(exc_info.value).lower()

    @patch("crewai.mcp.tool_resolver.asyncio.sleep", new_callable=AsyncMock)
    @patch("crewai.mcp.tool_resolver.MCPClient")
    def test_disconnect_runs_when_cancellation_occurs_after_connect(
        self, mock_client_class, _mock_sleep, resolver, http_config
    ):
        mock_client = AsyncMock()
        mock_client.connected = False

        async def _connect():
            mock_client.connected = True

        cancelled = asyncio.CancelledError()
        cancelled.__context__ = _http_status_error(401)

        mock_client.connect = AsyncMock(side_effect=_connect)
        mock_client.list_tools = AsyncMock(side_effect=cancelled)
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        with pytest.raises(MCPAuthenticationError) as exc_info:
            resolver._resolve_native(http_config)

        mock_client.disconnect.assert_awaited_once()
        assert exc_info.value.status_code == 401
        assert exc_info.value.__cause__ is cancelled


class TestAttemptMcpDiscoveryAuthErrors:
    @pytest.mark.asyncio
    async def test_attempt_mcp_discovery_reports_authentication_failure_for_401(self):
        async def _fail(_server_url: str) -> dict[str, dict[str, object]]:
            raise _http_status_error(401)

        result, error, should_retry = await MCPToolResolver._attempt_mcp_discovery(
            _fail, "https://mcp.example.com/api"
        )

        assert result is None
        assert should_retry is False
        assert "401 Unauthorized" in error
        assert "authentication failure" in error
