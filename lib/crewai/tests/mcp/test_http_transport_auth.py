"""Tests for HTTP transport authentication error handling."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from crewai.mcp.exceptions import MCPAuthenticationError, MCPHTTPError
from crewai.mcp.transports.http import HTTPTransport

if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mcp.example.com/mcp")
    response = httpx.Response(status_code, text="refused", request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


@pytest.mark.asyncio
async def test_http_transport_connect_raises_authentication_error_for_401():
    transport = HTTPTransport(
        url="https://mcp.example.com/mcp",
        headers={"Authorization": "Bearer stale-token"},
    )
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=_http_status_error(401))

    with patch(
        "mcp.client.streamable_http.streamablehttp_client",
        return_value=mock_context,
    ):
        with pytest.raises(MCPAuthenticationError) as exc_info:
            await transport.connect()

    assert exc_info.value.status_code == 401
    assert "401 Unauthorized" in str(exc_info.value)
    assert "authentication failure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_transport_connect_raises_authentication_error_for_403():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=_http_status_error(403))

    with patch(
        "mcp.client.streamable_http.streamablehttp_client",
        return_value=mock_context,
    ):
        with pytest.raises(MCPAuthenticationError) as exc_info:
            await transport.connect()

    assert exc_info.value.status_code == 403
    assert "403 Forbidden" in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_transport_connect_raises_http_error_for_non_auth_status():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=_http_status_error(500))

    with patch(
        "mcp.client.streamable_http.streamablehttp_client",
        return_value=mock_context,
    ):
        with pytest.raises(MCPHTTPError) as exc_info:
            await transport.connect()

    assert exc_info.value.status_code == 500
    assert "500 Internal Server Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_transport_connect_classifies_mixed_exception_group():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    auth_error = _http_status_error(401)
    cancelled = asyncio.CancelledError()
    mixed_group = BaseExceptionGroup("task group failed", [auth_error, cancelled])

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=mixed_group)

    with patch(
        "mcp.client.streamable_http.streamablehttp_client",
        return_value=mock_context,
    ):
        with pytest.raises(MCPAuthenticationError) as exc_info:
            await transport.connect()

    assert exc_info.value.status_code == 401
    assert transport._transport_context is None
    assert exc_info.value.__cause__ is mixed_group


@pytest.mark.asyncio
async def test_http_transport_connect_cancelled_on_enter_recovers_auth_on_exit():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    cancelled = asyncio.CancelledError()
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(side_effect=cancelled)
    mock_context.__aexit__ = AsyncMock(side_effect=_http_status_error(401))

    with patch(
        "mcp.client.streamable_http.streamablehttp_client",
        return_value=mock_context,
    ):
        with pytest.raises(MCPAuthenticationError) as exc_info:
            await transport.connect()

    mock_context.__aexit__.assert_awaited_once()
    assert exc_info.value.status_code == 401
    assert transport._transport_context is None


@pytest.mark.asyncio
async def test_http_transport_disconnect_recovers_auth_from_exit():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    mock_context = MagicMock()
    mock_context.__aexit__ = AsyncMock(side_effect=_http_status_error(401))
    transport._transport_context = mock_context
    transport._set_streams(MagicMock(), MagicMock())

    with pytest.raises(MCPAuthenticationError) as exc_info:
        await transport.disconnect()

    mock_context.__aexit__.assert_awaited_once()
    assert exc_info.value.status_code == 401
    assert transport._transport_context is None
    assert not transport.connected


@pytest.mark.asyncio
async def test_http_transport_disconnect_propagates_cancellation_without_http_status():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    mock_context = MagicMock()
    mock_context.__aexit__ = AsyncMock(side_effect=asyncio.CancelledError())

    transport._transport_context = mock_context
    transport._set_streams(MagicMock(), MagicMock())

    with pytest.raises(asyncio.CancelledError):
        await transport.disconnect()

    mock_context.__aexit__.assert_awaited_once()
    assert transport._transport_context is None
    assert not transport.connected


@pytest.mark.asyncio
async def test_http_transport_connect_timeout_recovers_auth_on_exit():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    mock_context = MagicMock()
    mock_context.__aexit__ = AsyncMock(side_effect=_http_status_error(401))

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            return_value=mock_context,
        ),
        patch(
            "crewai.mcp.transports.http.asyncio.wait_for",
            side_effect=asyncio.TimeoutError(),
        ),
    ):
        with pytest.raises(MCPAuthenticationError) as exc_info:
            await transport.connect()

    mock_context.__aexit__.assert_awaited_once()
    assert exc_info.value.status_code == 401
    assert transport._transport_context is None
