"""Tests for HTTP transport authentication error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from crewai.mcp.exceptions import MCPAuthenticationError, MCPHTTPError
from crewai.mcp.transports.http import HTTPTransport


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
