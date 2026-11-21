"""A tools/call that was sent must not be sent again after a timeout or error.

The server may already have run the tool (sent an email, placed an order), so
resending the request can run a non-idempotent tool more than once.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.mcp.client import MCPClient
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper

from tests.mcp.test_client_tool_execution_auth import MockTransport


@pytest.fixture(autouse=True)
def _no_backoff():
    with patch("asyncio.sleep", AsyncMock()):
        yield


def _connected_client(call_tool: AsyncMock) -> MCPClient:
    transport = MockTransport()
    transport._connected = True
    client = MCPClient(transport)
    client._initialized = True
    client._session = MagicMock(call_tool=call_tool)
    return client


@pytest.mark.asyncio
async def test_client_does_not_resend_tool_call_after_timeout():
    call_tool = AsyncMock(side_effect=asyncio.TimeoutError())
    client = _connected_client(call_tool)

    with pytest.raises(ConnectionError, match="timed out"):
        await client.call_tool_result("send_email", {"to": "a@example.com"})

    assert call_tool.await_count == 1


@pytest.mark.asyncio
async def test_client_does_not_resend_tool_call_after_transport_error():
    call_tool = AsyncMock(side_effect=RuntimeError("connection reset by peer"))
    client = _connected_client(call_tool)

    with pytest.raises(ConnectionError):
        await client.call_tool_result("send_email", {"to": "a@example.com"})

    assert call_tool.await_count == 1


@pytest.mark.asyncio
async def test_client_still_retries_tool_discovery():
    list_tools = AsyncMock(side_effect=asyncio.TimeoutError())
    client = _connected_client(AsyncMock())
    client._session.list_tools = list_tools

    with pytest.raises(ConnectionError):
        await client.list_tools(use_cache=False)

    assert list_tools.await_count == client.max_retries


def _wrapper() -> MCPToolWrapper:
    return MCPToolWrapper(
        mcp_server_params={"url": "https://mcp.example.com/mcp"},
        tool_name="send_email",
        tool_schema={"description": "Send an email"},
        server_name="mail",
    )


def _patch_mcp(session: MagicMock, connect_error: Exception | None = None):
    @asynccontextmanager
    async def fake_streamable_client(*args, **kwargs):
        if connect_error is not None:
            raise connect_error
        yield MagicMock(), MagicMock(), MagicMock()

    @asynccontextmanager
    async def fake_session(*args, **kwargs):
        yield session

    return (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            fake_streamable_client,
        ),
        patch("mcp.ClientSession", fake_session),
    )


@pytest.mark.parametrize(
    "error",
    [asyncio.TimeoutError(), RuntimeError("connection reset by peer")],
)
def test_wrapper_does_not_resend_tool_call_after_it_was_sent(error: Exception):
    session = MagicMock(
        initialize=AsyncMock(), call_tool=AsyncMock(side_effect=error)
    )
    client_patch, session_patch = _patch_mcp(session)

    with client_patch, session_patch:
        result = _wrapper().run()

    assert session.call_tool.await_count == 1
    assert "may already have run it" in result


def test_wrapper_still_retries_when_connecting_fails():
    session = MagicMock(initialize=AsyncMock(), call_tool=AsyncMock())
    client_patch, session_patch = _patch_mcp(
        session, connect_error=RuntimeError("connection refused")
    )

    with client_patch, session_patch:
        result = _wrapper().run()

    assert "failed after 3 attempts" in result
    session.call_tool.assert_not_awaited()
