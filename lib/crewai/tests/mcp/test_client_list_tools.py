"""Tests for MCP tool discovery pagination."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from crewai.mcp.client import MCPClient
from crewai.mcp.transports.base import BaseTransport, TransportType


class MockTransport(BaseTransport):
    @property
    def transport_type(self) -> TransportType:
        return TransportType.STREAMABLE_HTTP

    async def connect(self) -> "MockTransport":
        self._connected = True
        return self

    async def disconnect(self) -> None:
        self._connected = False

    async def __aenter__(self) -> "MockTransport":
        return await self.connect()

    async def __aexit__(self, *args: object) -> None:
        await self.disconnect()


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"Description for {name}",
        inputSchema={"type": "object"},
    )


@pytest.mark.asyncio
async def test_list_tools_follows_mcp_pagination_cursor() -> None:
    client = MCPClient(MockTransport())
    session = MagicMock()
    session.list_tools = AsyncMock(
        side_effect=[
            SimpleNamespace(tools=[_tool("first-tool")], nextCursor="opaque/+=cursor"),
            SimpleNamespace(tools=[_tool("second-tool")], nextCursor=None),
        ]
    )
    client._session = session
    client._initialized = True

    tools = await client._list_tools_impl()

    assert [tool["original_name"] for tool in tools] == [
        "first-tool",
        "second-tool",
    ]
    assert session.list_tools.call_args_list[0].args == ()
    assert session.list_tools.call_args_list[1].kwargs == {
        "cursor": "opaque/+=cursor"
    }
