"""Regression tests for paginated MCP tool discovery."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

from crewai.mcp.client import MCPClient
from crewai.mcp.exceptions import MCPConnectionError
from crewai.mcp.transports.base import BaseTransport, TransportType
from mcp import ClientSession
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ListToolsRequest, ListToolsResult, PaginatedRequestParams, Tool
import pytest


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MCPClient:
    monkeypatch.setattr("crewai.mcp.client._mcp_schema_cache", {})
    transport = MagicMock(spec=BaseTransport)
    transport.connected = True
    transport.transport_type = TransportType.STREAMABLE_HTTP
    instance = MCPClient(transport, max_retries=1, cache_tools_list=True)
    instance._initialized = True
    instance._session = AsyncMock(spec=ClientSession)
    return instance


def tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Description of {name}",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


@pytest.mark.asyncio
async def test_single_page_preserves_tool_metadata(client: MCPClient) -> None:
    advertised = tool("search-tickets")
    client.session.list_tools.return_value = ListToolsResult(tools=[advertised])

    assert await client.list_tools() == [
        {
            "name": "search_tickets",
            "original_name": advertised.name,
            "description": advertised.description,
            "inputSchema": advertised.inputSchema,
        }
    ]
    assert client.session.list_tools.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["opaque:next-page/+=token", ""])
async def test_discovers_all_pages_and_preserves_opaque_cursor(
    client: MCPClient, cursor: str
) -> None:
    client.session.list_tools.side_effect = [
        ListToolsResult(tools=[tool("search-tickets")], nextCursor=cursor),
        ListToolsResult(tools=[tool("read-ticket")]),
    ]

    discovered = await client.list_tools()

    assert [item["original_name"] for item in discovered] == [
        "search-tickets",
        "read-ticket",
    ]
    assert [item["name"] for item in discovered] == ["search_tickets", "read_ticket"]
    assert client.session.list_tools.await_count == 2
    assert client.session.list_tools.await_args_list[1] == call(
        params=PaginatedRequestParams(cursor=cursor)
    )


@pytest.mark.asyncio
async def test_empty_page_with_cursor_does_not_end_discovery(client: MCPClient) -> None:
    client.session.list_tools.side_effect = [
        ListToolsResult(tools=[], nextCursor="page-2"),
        ListToolsResult(tools=[tool("read-ticket")], nextCursor="page-3"),
        ListToolsResult(tools=[]),
    ]

    assert [item["name"] for item in await client.list_tools()] == ["read_ticket"]
    assert client.session.list_tools.await_count == 3


@pytest.mark.asyncio
async def test_empty_unpaginated_catalog_returns_empty_list(client: MCPClient) -> None:
    client.session.list_tools.return_value = ListToolsResult(tools=[])

    assert await client.list_tools() == []
    assert client.session.list_tools.await_count == 1


@pytest.mark.asyncio
async def test_cache_contains_complete_catalog_and_can_be_bypassed(
    client: MCPClient,
) -> None:
    client.session.list_tools.side_effect = [
        ListToolsResult(tools=[tool("search")], nextCursor="page-2"),
        ListToolsResult(tools=[tool("read")]),
        ListToolsResult(tools=[tool("updated-search")], nextCursor="new-page-2"),
        ListToolsResult(tools=[tool("updated-read")]),
    ]

    first = await client.list_tools()
    cached = await client.list_tools()
    assert [item["name"] for item in cached] == ["search", "read"]
    assert cached == first
    assert client.session.list_tools.await_count == 2

    fresh = await client.list_tools(use_cache=False)
    assert [item["name"] for item in fresh] == ["updated_search", "updated_read"]
    assert client.session.list_tools.await_count == 4


@pytest.mark.asyncio
async def test_later_page_failure_does_not_return_or_cache_partial_catalog(
    client: MCPClient,
) -> None:
    client.session.list_tools.side_effect = [
        ListToolsResult(tools=[tool("old-first")], nextCursor="page-2"),
        MCPConnectionError("server disconnected"),
        ListToolsResult(tools=[tool("new-first")], nextCursor="new-page-2"),
        ListToolsResult(tools=[tool("new-second")]),
    ]

    with pytest.raises(MCPConnectionError, match="server disconnected"):
        await client.list_tools()

    assert [item["name"] for item in await client.list_tools()] == [
        "new_first",
        "new_second",
    ]
    assert client.session.list_tools.await_count == 4


@pytest.mark.asyncio
async def test_retry_restarts_discovery_without_duplicating_first_page(
    client: MCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.max_retries = 2
    monkeypatch.setattr("crewai.mcp.client.asyncio.sleep", AsyncMock())
    client.session.list_tools.side_effect = [
        ListToolsResult(tools=[tool("first")], nextCursor="page-2"),
        RuntimeError("temporary server failure"),
        ListToolsResult(tools=[tool("first")], nextCursor="page-2"),
        ListToolsResult(tools=[tool("second")]),
    ]

    assert [item["name"] for item in await client.list_tools()] == ["first", "second"]
    assert client.session.list_tools.await_count == 4
    assert (
        client.session.list_tools.await_args_list[0]
        == client.session.list_tools.await_args_list[2]
    )


@pytest.mark.asyncio
async def test_cancellation_during_later_page_propagates_without_caching(
    client: MCPClient,
) -> None:
    reached_second_page = asyncio.Event()
    page_cancelled = asyncio.Event()
    requests = 0

    async def list_page(
        *, params: PaginatedRequestParams | None = None
    ) -> ListToolsResult:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ListToolsResult(tools=[tool("first")], nextCursor="page-2")
        reached_second_page.set()
        try:
            return await asyncio.Future[ListToolsResult]()
        finally:
            page_cancelled.set()

    client.session.list_tools.side_effect = list_page
    discovery = asyncio.create_task(client.list_tools())
    try:
        await asyncio.wait_for(reached_second_page.wait(), timeout=1)
    finally:
        discovery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await discovery

    assert page_cancelled.is_set()
    assert requests == 2
    client.session.list_tools.side_effect = None
    client.session.list_tools.return_value = ListToolsResult(tools=[tool("fresh")])
    assert [item["name"] for item in await client.list_tools()] == ["fresh"]


@pytest.mark.asyncio
async def test_stalled_later_page_obeys_discovery_timeout(client: MCPClient) -> None:
    client.discovery_timeout = 1
    cancelled = asyncio.Event()
    requests = 0

    async def list_page(
        *, params: PaginatedRequestParams | None = None
    ) -> ListToolsResult:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ListToolsResult(tools=[tool("first")], nextCursor="page-2")
        try:
            return await asyncio.Future[ListToolsResult]()
        finally:
            cancelled.set()

    client.session.list_tools.side_effect = list_page
    with pytest.raises(ConnectionError, match="timed out"):
        await asyncio.wait_for(client.list_tools(), timeout=3)

    assert requests == 2
    assert cancelled.is_set()
    client.session.list_tools.side_effect = None
    client.session.list_tools.return_value = ListToolsResult(tools=[tool("fresh")])
    assert [item["name"] for item in await client.list_tools()] == ["fresh"]


@pytest.mark.asyncio
async def test_pagination_through_real_mcp_client_and_server(client: MCPClient) -> None:
    server = Server("paginated-tools")
    requested_cursors: list[str | None] = []

    @server.list_tools()
    async def list_page(request: ListToolsRequest) -> ListToolsResult:
        cursor = request.params.cursor if request.params else None
        requested_cursors.append(cursor)
        if cursor is None:
            return ListToolsResult(tools=[tool("search")], nextCursor="opaque:+/=")
        assert cursor == "opaque:+/="
        return ListToolsResult(tools=[tool("read")])

    async with create_connected_server_and_client_session(server) as session:
        client._session = session
        discovered = await client.list_tools()

    assert [item["name"] for item in discovered] == ["search", "read"]
    assert requested_cursors == [None, "opaque:+/="]
