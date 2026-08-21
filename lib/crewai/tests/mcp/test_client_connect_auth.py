"""Tests for MCPClient connect authentication error handling."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.mcp_events import MCPConnectionFailedEvent
from crewai.mcp.client import MCPClient
from crewai.mcp.exceptions import MCPAuthenticationError
from crewai.mcp.transports.base import BaseTransport, TransportType
from crewai.mcp.transports.http import HTTPTransport


class MockTransport(BaseTransport):
    """Minimal transport stub for connect() error-path tests."""

    @property
    def transport_type(self) -> TransportType:
        return TransportType.STREAMABLE_HTTP

    async def connect(self) -> "MockTransport":
        self._read_stream = MagicMock()
        self._write_stream = MagicMock()
        self._connected = True
        return self

    async def disconnect(self) -> None:
        self._connected = False

    async def __aenter__(self) -> "MockTransport":
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.disconnect()


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mcp.example.com/mcp")
    response = httpx.Response(status_code, text="refused", request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


@pytest.mark.asyncio
async def test_connect_cancelled_with_auth_status_in_cleanup():
    transport = MockTransport()
    client = MCPClient(transport)
    auth_error = _http_status_error(401)
    failed_events: list[MCPConnectionFailedEvent] = []

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock(side_effect=asyncio.CancelledError())
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("mcp.ClientSession", return_value=mock_session),
        patch.object(
            client,
            "_cleanup_on_error",
            AsyncMock(return_value=auth_error),
        ),
        crewai_event_bus.scoped_handlers(),
    ):
        @crewai_event_bus.on(MCPConnectionFailedEvent)
        def _capture(_: object, event: MCPConnectionFailedEvent) -> None:
            failed_events.append(event)

        with pytest.raises(MCPAuthenticationError) as exc_info:
            await client.connect()

        assert crewai_event_bus.flush(timeout=10)

    assert exc_info.value.status_code == 401
    assert len(failed_events) == 1
    assert failed_events[0].error_type == "authentication"
    assert failed_events[0].status_code == 401
    assert "401 Unauthorized" in failed_events[0].error


@pytest.mark.asyncio
async def test_connect_cancelled_without_underlying_failure_emits_cancelled():
    transport = MockTransport()
    client = MCPClient(transport)
    failed_events: list[MCPConnectionFailedEvent] = []

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock(side_effect=asyncio.CancelledError())
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("mcp.ClientSession", return_value=mock_session),
        patch.object(client, "_cleanup_on_error", AsyncMock(return_value=None)),
        crewai_event_bus.scoped_handlers(),
    ):
        @crewai_event_bus.on(MCPConnectionFailedEvent)
        def _capture(_: object, event: MCPConnectionFailedEvent) -> None:
            failed_events.append(event)

        with pytest.raises(asyncio.CancelledError):
            await client.connect()

        assert crewai_event_bus.flush(timeout=10)

    assert len(failed_events) == 1
    assert failed_events[0].error_type == "cancelled"
    assert failed_events[0].error == "Connection cancelled"


@pytest.mark.asyncio
async def test_connect_raises_authentication_error_for_typed_transport_failure():
    transport = MockTransport()
    client = MCPClient(transport)
    failed_events: list[MCPConnectionFailedEvent] = []

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock(
        side_effect=MCPAuthenticationError(401)
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("mcp.ClientSession", return_value=mock_session),
        patch.object(
            client,
            "_cleanup_on_error",
            AsyncMock(return_value=None),
        ),
        crewai_event_bus.scoped_handlers(),
    ):
        @crewai_event_bus.on(MCPConnectionFailedEvent)
        def _capture(_: object, event: MCPConnectionFailedEvent) -> None:
            failed_events.append(event)

        with pytest.raises(MCPAuthenticationError):
            await client.connect()

        assert crewai_event_bus.flush(timeout=10)

    assert len(failed_events) == 1
    assert failed_events[0].error_type == "authentication"


@pytest.mark.asyncio
async def test_connect_cancelled_during_initialize_recovers_auth_on_transport_unwind():
    transport = HTTPTransport(url="https://mcp.example.com/mcp")
    client = MCPClient(transport)
    auth_error = _http_status_error(401)
    failed_events: list[MCPConnectionFailedEvent] = []

    mock_streams = (MagicMock(), MagicMock(), None)
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_streams)
    mock_context.__aexit__ = AsyncMock(side_effect=auth_error)

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock(side_effect=asyncio.CancelledError())
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            return_value=mock_context,
        ),
        patch("mcp.ClientSession", return_value=mock_session),
        crewai_event_bus.scoped_handlers(),
    ):
        @crewai_event_bus.on(MCPConnectionFailedEvent)
        def _capture(_: object, event: MCPConnectionFailedEvent) -> None:
            failed_events.append(event)

        with pytest.raises(MCPAuthenticationError) as exc_info:
            await client.connect()

        assert crewai_event_bus.flush(timeout=10)

    mock_context.__aexit__.assert_awaited()
    assert exc_info.value.status_code == 401
    assert len(failed_events) == 1
    assert failed_events[0].error_type == "authentication"
    assert failed_events[0].status_code == 401


@pytest.mark.asyncio
async def test_disconnect_preserves_authentication_error_from_teardown():
    transport = MockTransport()
    client = MCPClient(transport)
    auth_error = MCPAuthenticationError(401)
    transport._connected = True
    client._initialized = True

    with patch.object(
        client._exit_stack, "aclose", AsyncMock(side_effect=auth_error)
    ):
        with pytest.raises(MCPAuthenticationError) as exc_info:
            await client.disconnect()

    assert exc_info.value is auth_error
    assert not client.connected
