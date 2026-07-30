"""MCP client connection lifecycle and retry-safety tests."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.mcp.client import MCPClient
from crewai.mcp.transports.base import BaseTransport, TransportType
from crewai.mcp.transports.http import HTTPTransport


class ConnectedTransport(BaseTransport):
    """Minimal connected transport for client operation tests."""

    @property
    def transport_type(self) -> TransportType:
        return TransportType.HTTP

    async def connect(self) -> ConnectedTransport:
        self._connected = True
        return self

    async def disconnect(self) -> None:
        self._connected = False

    async def __aenter__(self) -> ConnectedTransport:
        return await self.connect()

    async def __aexit__(self, *_args: Any) -> None:
        await self.disconnect()


class FailingSession:
    """Session that models a lost response after a side effect committed."""

    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, _name: str, _arguments: dict[str, Any]) -> None:
        self.calls += 1
        raise ConnectionError("connection dropped after commit")


@pytest.mark.asyncio
async def test_connect_timeout_bounds_transport_startup():
    """connect_timeout must include transport context entry."""

    class HangingContext(AbstractAsyncContextManager):
        async def __aenter__(self):
            await asyncio.Event().wait()

        async def __aexit__(self, *_args: Any):
            return None

    client = MCPClient(
        HTTPTransport("https://mcp.example.com"),
        connect_timeout=1,
    )

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            return_value=HangingContext(),
        ),
        patch.object(crewai_event_bus, "emit"),
        pytest.raises(ConnectionError, match="timed out after 1 seconds"),
    ):
        await asyncio.wait_for(client.connect(), timeout=1.25)


@pytest.mark.asyncio
async def test_http_transport_context_enters_and_exits_in_same_task():
    """AnyIO transport cancel scopes must not cross asyncio tasks."""
    entered_task = None
    exited_task = None

    class RecordingContext(AbstractAsyncContextManager):
        async def __aenter__(self):
            nonlocal entered_task
            entered_task = asyncio.current_task()
            return MagicMock(), MagicMock(), None

        async def __aexit__(self, *_args: Any):
            nonlocal exited_task
            exited_task = asyncio.current_task()

    transport = HTTPTransport("https://mcp.example.com")
    with patch(
        "mcp.client.streamable_http.streamablehttp_client",
        return_value=RecordingContext(),
    ):
        await transport.connect()
        await transport.disconnect()

    assert entered_task is exited_task


@pytest.mark.asyncio
async def test_tool_call_is_not_retried_by_default():
    """A lost response must not blindly replay a potentially mutating call."""
    transport = ConnectedTransport()
    await transport.connect()
    client = MCPClient(transport, max_retries=3)
    session = FailingSession()
    client._session = session
    client._initialized = True

    with (
        patch.object(crewai_event_bus, "emit"),
        pytest.raises(ConnectionError, match="connection dropped after commit"),
    ):
        await client.call_tool("mutating_tool")

    assert session.calls == 1


@pytest.mark.asyncio
async def test_tool_call_retries_require_explicit_opt_in():
    """Callers can explicitly accept replay risk for idempotent tools."""
    transport = ConnectedTransport()
    await transport.connect()
    client = MCPClient(
        transport,
        max_retries=3,
        retry_tool_calls=True,
    )
    session = FailingSession()
    client._session = session
    client._initialized = True

    with (
        patch.object(crewai_event_bus, "emit"),
        patch("crewai.mcp.client.asyncio.sleep", new=AsyncMock()),
        pytest.raises(ConnectionError, match="failed after 3 attempts"),
    ):
        await client.call_tool("idempotent_tool")

    assert session.calls == 3


@pytest.mark.asyncio
async def test_cleanup_failure_is_logged_without_masking_connection_error():
    """Best-effort cleanup must preserve the original connection exception."""
    logger = MagicMock(spec=logging.Logger)
    client = MCPClient(ConnectedTransport(), logger=logger)

    async def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    client._exit_stack.push_async_callback(fail_cleanup)
    await client._cleanup_on_error()

    logger.warning.assert_called_once()
    log_template, log_error = logger.warning.call_args.args
    assert log_template == "Error during MCP client cleanup: %s"
    assert str(log_error) == "cleanup failed"
