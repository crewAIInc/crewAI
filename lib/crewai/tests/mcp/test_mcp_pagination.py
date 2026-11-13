"""Regression tests for MCP discovery pagination (issue #8216).

MCP servers may split ``prompts/list`` and ``tools/list`` across pages.
Discovery must follow every returned ``nextCursor`` instead of silently
returning only the first page.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import mcp
import mcp.client.streamable_http
from crewai.mcp.client import MCPClient
from crewai.mcp.tool_resolver import MCPToolResolver
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


def _paged_result(items, next_cursor):
    return SimpleNamespace(items=items, nextCursor=next_cursor)


class PagedPromptSession:
    """Fake MCP session serving two prompts, one per page."""

    def __init__(self):
        self.cursors_seen: list = []

    async def list_prompts(self, cursor=None):
        self.cursors_seen.append(cursor)
        if cursor is None:
            prompts = [SimpleNamespace(name="alpha", description="a", arguments=[])]
            return SimpleNamespace(prompts=prompts, nextCursor="cursor-p2")
        prompts = [SimpleNamespace(name="beta", description="b", arguments=[])]
        return SimpleNamespace(prompts=prompts, nextCursor=None)


@pytest.mark.asyncio
async def test_list_prompts_follows_all_pages():
    client = MCPClient(MockTransport())
    session = PagedPromptSession()
    client._session = session

    prompts = await client._list_prompts_impl()

    assert session.cursors_seen == [None, "cursor-p2"]
    assert [p["name"] for p in prompts] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_list_prompts_single_page_unaffected():
    client = MCPClient(MockTransport())

    async def one_page(cursor=None):
        prompts = [SimpleNamespace(name="only", description="", arguments=[])]
        return SimpleNamespace(prompts=prompts, nextCursor=None)

    client._session = SimpleNamespace(list_prompts=one_page)

    prompts = await client._list_prompts_impl()

    assert [p["name"] for p in prompts] == ["only"]


class PagedToolSession:
    """Fake MCP session serving two tools, one per page, for the
    streamable-HTTP discovery path."""

    def __init__(self):
        self.cursors_seen: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return SimpleNamespace()

    async def list_tools(self, cursor=None):
        self.cursors_seen.append(cursor)
        tool = SimpleNamespace(
            name="page_two_tool" if cursor else "page_one_tool",
            description="t",
            inputSchema={"type": "object", "properties": {}},
        )
        next_cursor = None if cursor else "cursor-2"
        return SimpleNamespace(tools=[tool], nextCursor=next_cursor)


@asynccontextmanager
async def _fake_streamable_http(url):
    yield (None, None, "fake-session-id")


@pytest.mark.asyncio
async def test_discover_mcp_tools_follows_all_pages():
    session = PagedToolSession()
    resolver = MCPToolResolver(agent=MagicMock(), logger=MagicMock())

    with (
        patch.object(mcp, "ClientSession", lambda *a, **k: session),
        patch.object(
            mcp.client.streamable_http, "streamablehttp_client", _fake_streamable_http
        ),
    ):
        schemas = await resolver._discover_mcp_tools("https://fake-host/mcp")

    assert session.cursors_seen == [None, "cursor-2"]
    assert sorted(schemas) == ["page_one_tool", "page_two_tool"]
