"""Tests for MCPClient tool execution authentication error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.mcp_events import MCPToolExecutionFailedEvent
from crewai.mcp.client import MCPClient
from crewai.mcp.exceptions import MCPAuthenticationError
from crewai.mcp.transports.base import BaseTransport, TransportType


class MockTransport(BaseTransport):
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

    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()


@pytest.mark.asyncio
async def test_call_tool_result_emits_authentication_error_type():
    transport = MockTransport()
    transport._connected = True
    client = MCPClient(transport)
    client._initialized = True
    failed_events: list[MCPToolExecutionFailedEvent] = []

    with (
        patch.object(
            client,
            "_retry_operation",
            AsyncMock(side_effect=MCPAuthenticationError(401)),
        ),
        crewai_event_bus.scoped_handlers(),
    ):
        @crewai_event_bus.on(MCPToolExecutionFailedEvent)
        def _capture(_: object, event: MCPToolExecutionFailedEvent) -> None:
            failed_events.append(event)

        with pytest.raises(MCPAuthenticationError):
            await client.call_tool_result("search", {"query": "test"})

        assert crewai_event_bus.flush(timeout=10)

    assert len(failed_events) == 1
    assert failed_events[0].error_type == "authentication"
    assert failed_events[0].tool_name == "search"
    assert "401 Unauthorized" in failed_events[0].error
