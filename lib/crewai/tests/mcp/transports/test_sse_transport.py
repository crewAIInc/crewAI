import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from crewai.mcp.transports.sse import SSETransport


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "storage"))


class FakeSSEClient:
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers
        self._read = AsyncMock()
        self._write = AsyncMock()

    async def __aenter__(self):
        return (self._read, self._write)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_mcp_sse():
    fake_mcp = ModuleType("mcp")
    fake_mcp_client = ModuleType("mcp.client")
    fake_mcp_client_sse = ModuleType("mcp.client.sse")

    sys.modules["mcp"] = fake_mcp
    sys.modules["mcp.client"] = fake_mcp_client
    sys.modules["mcp.client.sse"] = fake_mcp_client_sse

    mock_sse_client = MagicMock(side_effect=FakeSSEClient)
    fake_mcp_client_sse.sse_client = mock_sse_client

    yield mock_sse_client

    del sys.modules["mcp.client.sse"]
    del sys.modules["mcp.client"]
    del sys.modules["mcp"]


@pytest.mark.asyncio
async def test_sse_transport_connect_without_terminate_on_close(mock_mcp_sse):
    transport = SSETransport(
        url="https://example.com/sse",
        headers={"Authorization": "Bearer token"},
    )

    await transport.connect()

    mock_mcp_sse.assert_called_once_with(
        "https://example.com/sse",
        headers={"Authorization": "Bearer token"},
    )

    call_kwargs = mock_mcp_sse.call_args[1]
    assert "terminate_on_close" not in call_kwargs

    assert transport._connected is True


@pytest.mark.asyncio
async def test_sse_transport_connect_without_headers(mock_mcp_sse):
    transport = SSETransport(url="https://example.com/sse")

    await transport.connect()

    mock_mcp_sse.assert_called_once_with(
        "https://example.com/sse",
        headers=None,
    )

    call_kwargs = mock_mcp_sse.call_args[1]
    assert "terminate_on_close" not in call_kwargs


@pytest.mark.asyncio
async def test_sse_transport_connect_sets_streams(mock_mcp_sse):
    transport = SSETransport(url="https://example.com/sse")

    await transport.connect()

    assert transport._read_stream is not None
    assert transport._write_stream is not None
    assert transport._connected is True


@pytest.mark.asyncio
async def test_sse_transport_context_manager(mock_mcp_sse):
    async with SSETransport(url="https://example.com/sse") as transport:
        assert transport._connected is True

    assert transport._connected is False


@pytest.mark.asyncio
async def test_sse_transport_connect_failure_raises_connection_error(mock_mcp_sse):
    mock_sse_client_error = MagicMock(
        side_effect=Exception("Connection failed")
    )

    fake_mcp_client_sse = sys.modules["mcp.client.sse"]
    fake_mcp_client_sse.sse_client = mock_sse_client_error

    transport = SSETransport(url="https://example.com/sse")

    with pytest.raises(ConnectionError, match="Failed to connect to SSE MCP server"):
        await transport.connect()

    assert transport._connected is False
