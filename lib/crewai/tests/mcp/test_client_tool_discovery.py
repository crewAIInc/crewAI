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
    """Create a connected client with an isolated schema cache and SDK mock."""
    monkeypatch.setattr("crewai.mcp.client._mcp_schema_cache", {})
    transport = MagicMock(spec=BaseTransport)
    transport.connected = True
    transport.transport_type = TransportType.STREAMABLE_HTTP
    instance = MCPClient(transport, max_retries=1, cache_tools_list=True)
    instance._initialized = True
    instance._session = AsyncMock(spec=ClientSession)
    return instance


def tool(name: str) -> Tool:
    """Build an advertised tool with metadata to verify across discovery pages."""
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
    """Keep tool metadata and name normalization unchanged for a single page."""
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
    """Pass cursor strings back unchanged, including an empty-string cursor."""
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
    """Continue after an empty page when the server advertises another cursor."""
    client.session.list_tools.side_effect = [
        ListToolsResult(tools=[], nextCursor="page-2"),
        ListToolsResult(tools=[tool("read-ticket")], nextCursor="page-3"),
        ListToolsResult(tools=[]),
    ]

    assert [item["name"] for item in await client.list_tools()] == ["read_ticket"]
    assert client.session.list_tools.await_count == 3


@pytest.mark.asyncio
async def test_empty_unpaginated_catalog_returns_empty_list(client: MCPClient) -> None:
    """Return an empty catalog when the first response has no continuation."""
    client.session.list_tools.return_value = ListToolsResult(tools=[])

    assert await client.list_tools() == []
    assert client.session.list_tools.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cursors",
    [
        ["page-2", "page-2"],
        ["", ""],
        ["page-2", "page-3", "page-2"],
    ],
    ids=["repeated", "empty", "cycle"],
)
async def test_repeated_cursor_fails_without_caching_partial_catalog(
    client: MCPClient, cursors: list[str]
) -> None:
    """Stop a cursor cycle before requesting it again or caching partial tools."""
    client.session.list_tools.side_effect = [
        *[
            ListToolsResult(tools=[tool(f"page-{index}")], nextCursor=cursor)
            for index, cursor in enumerate(cursors)
        ],
        ListToolsResult(tools=[tool("unexpected-extra-page")]),
    ]

    with pytest.raises(ConnectionError, match="repeated tools/list pagination cursor"):
        await client.list_tools()

    assert client.session.list_tools.await_count == len(cursors)
    client.session.list_tools.side_effect = None
    client.session.list_tools.return_value = ListToolsResult(tools=[tool("fresh")])
    assert [item["name"] for item in await client.list_tools()] == ["fresh"]


@pytest.mark.asyncio
async def test_retry_after_repeated_cursor_starts_with_fresh_state(
    client: MCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow a retry to reuse a cursor without retaining failed-attempt tools."""
    client.max_retries = 2
    monkeypatch.setattr("crewai.mcp.client.asyncio.sleep", AsyncMock())
    client.session.list_tools.side_effect = [
        ListToolsResult(tools=[tool("old-first")], nextCursor="page-2"),
        ListToolsResult(tools=[tool("old-second")], nextCursor="page-2"),
        ListToolsResult(tools=[tool("new-first")], nextCursor="page-2"),
        ListToolsResult(tools=[tool("new-second")]),
    ]

    discovered = await client.list_tools()

    assert [item["name"] for item in discovered] == ["new_first", "new_second"]
    assert client.session.list_tools.await_args_list == [
        call(params=None),
        call(params=PaginatedRequestParams(cursor="page-2")),
        call(params=None),
        call(params=PaginatedRequestParams(cursor="page-2")),
    ]
    assert await client.list_tools() == discovered
    assert client.session.list_tools.await_count == 4


@pytest.mark.asyncio
async def test_cache_contains_complete_catalog_and_can_be_bypassed(
    client: MCPClient,
) -> None:
    """Reuse complete cached catalogs and fetch every page on explicit bypass."""
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
    """Discard partial tools after failure so the next call discovers afresh."""
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
    """Restart at the first page without retaining tools from a failed attempt."""
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
    """Propagate cancellation into the pending page and discard partial tools."""
    reached_second_page = asyncio.Event()
    page_cancelled = asyncio.Event()
    requests = 0

    async def list_page(
        *, params: PaginatedRequestParams | None = None
    ) -> ListToolsResult:
        """Signal when the second page is awaiting cancellation from the caller."""
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
    """Cancel a stalled later page within the existing discovery timeout."""
    client.discovery_timeout = 1
    cancelled = asyncio.Event()
    requests = 0

    async def list_page(
        *, params: PaginatedRequestParams | None = None
    ) -> ListToolsResult:
        """Stall after the first page and record cancellation by the timeout."""
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
    """Verify cursor exchange and complete discovery through real SDK sessions."""
    server = Server("paginated-tools")
    requested_cursors: list[str | None] = []

    @server.list_tools()
    async def list_page(request: ListToolsRequest) -> ListToolsResult:
        """Advertise two pages and record the cursors received over the protocol."""
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
