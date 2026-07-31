"""MCP client connection lifecycle and retry-safety tests."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
import logging
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.mcp_events import MCPConnectionFailedEvent
from crewai.mcp.client import MCPClient
from crewai.mcp.transports.base import BaseTransport, TransportType
from crewai.mcp.transports.http import HTTPTransport


if sys.version_info >= (3, 11):
    from builtins import BaseExceptionGroup
else:
    from exceptiongroup import BaseExceptionGroup


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


class LifecycleTransport(ConnectedTransport):
    """Connected transport with placeholder streams for session startup tests."""

    async def connect(self) -> LifecycleTransport:
        self._set_streams(MagicMock(), MagicMock())
        return self


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


@pytest.mark.asyncio
async def test_cancelled_startup_reports_nested_tls_error():
    """A transport task failure must beat the cancellation used to signal it."""

    class TLSFailingTransportContext(AbstractAsyncContextManager):
        async def __aenter__(self):
            return MagicMock(), MagicMock(), None

        async def __aexit__(self, exc_type, _exc, _tb):
            assert exc_type is asyncio.CancelledError
            raise BaseExceptionGroup(
                "transport failed",
                [
                    ConnectionError(
                        "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer"
                    ),
                    GeneratorExit(),
                ],
            )

    class TLSFailingSession:
        async def __aenter__(self):
            return self

        async def initialize(self) -> None:
            raise asyncio.CancelledError

        async def __aexit__(self, *_args):
            return None

    emitted_events = []
    client = MCPClient(HTTPTransport("https://mcp.example.com"))

    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            return_value=TLSFailingTransportContext(),
        ),
        patch("mcp.ClientSession", return_value=TLSFailingSession()),
        patch.object(
            crewai_event_bus,
            "emit",
            side_effect=lambda _source, event: emitted_events.append(event),
        ),
        pytest.raises(ConnectionError, match="CERTIFICATE_VERIFY_FAILED"),
    ):
        await client.connect()

    failed_event = next(
        event for event in emitted_events if isinstance(event, MCPConnectionFailedEvent)
    )
    assert failed_event.error_type == "tls"
    assert "CERTIFICATE_VERIFY_FAILED" in failed_event.error


@pytest.mark.asyncio
async def test_external_startup_cancellation_remains_cancelled():
    """Cancellation without an underlying transport error must propagate."""

    class CancelledSession:
        async def __aenter__(self):
            return self

        async def initialize(self) -> None:
            raise asyncio.CancelledError

        async def __aexit__(self, *_args):
            return None

    emitted_events = []
    client = MCPClient(LifecycleTransport())

    with (
        patch("mcp.ClientSession", return_value=CancelledSession()),
        patch.object(
            crewai_event_bus,
            "emit",
            side_effect=lambda _source, event: emitted_events.append(event),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await client.connect()

    failed_event = next(
        event for event in emitted_events if isinstance(event, MCPConnectionFailedEvent)
    )
    assert failed_event.error_type == "cancelled"
