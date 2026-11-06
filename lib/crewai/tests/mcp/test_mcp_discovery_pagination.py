"""Pagination coverage for the MCP discovery paths the tools fix does not touch.

An MCP server may split ``prompts/list`` and ``tools/list`` across pages and hand
back a ``nextCursor``. Three call sites fetch listings in CrewAI's MCP layer:

* ``MCPClient._list_tools_impl`` (native client) - reported as #8098.
* ``MCPClient._list_prompts_impl`` (native client) - covered here.
* ``MCPToolResolver._discover_mcp_tools`` (HTTPS/streamable-HTTP discovery) -
  covered here.

Each test fails on a single-page implementation and passes once every returned
``nextCursor`` is followed.
"""

import logging
import sys

import pytest

PAGINATED_SERVER = '''
import json
import sys

PROMPTS = ["alpha", "beta"]
TOOLS = ["tool_a", "tool_b"]
REPEAT_CURSOR = "--repeat-cursor" in sys.argv


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


def prompt_obj(name):
    return {"name": name, "description": f"{name} prompt", "arguments": []}


def tool_obj(name):
    return {
        "name": name,
        "description": f"{name} tool",
        "inputSchema": {"type": "object", "properties": {}},
    }


def page(items, builder, cursor, next_cursor):
    if REPEAT_CURSOR:
        return {"items": [builder(items[0])], "nextCursor": next_cursor}
    if cursor is None:
        return {"items": [builder(items[0])], "nextCursor": next_cursor}
    if cursor == next_cursor:
        return {"items": [builder(items[1])]}
    return None


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    mid = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"prompts": {}, "tools": {}},
            "serverInfo": {"name": "paged-server", "version": "1.0.0"},
        }})
    elif method == "prompts/list":
        result = page(PROMPTS, prompt_obj, params.get("cursor"), "cursor-p2")
        payload = {"prompts": result["items"]}
        if "nextCursor" in result:
            payload["nextCursor"] = result["nextCursor"]
        send({"jsonrpc": "2.0", "id": mid, "result": payload})
    elif method == "tools/list":
        result = page(TOOLS, tool_obj, params.get("cursor"), "cursor-t2")
        payload = {"tools": result["items"]}
        if "nextCursor" in result:
            payload["nextCursor"] = result["nextCursor"]
        send({"jsonrpc": "2.0", "id": mid, "result": payload})
    elif method is not None and method.startswith("notifications/"):
        pass
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid,
              "error": {"code": -32601, "message": f"method not found: {method}"}})
'''


@pytest.fixture
def paged_server(tmp_path):
    """A real stdio MCP server that paginates prompts/list and tools/list."""
    server = tmp_path / "paged_server.py"
    server.write_text(PAGINATED_SERVER, encoding="utf-8")
    return str(server)


@pytest.mark.asyncio
async def test_list_prompts_follows_pagination(paged_server):
    """Regression guard: list_prompts() returned only the first page.

    The server advertises two prompts across two pages; the native client has to
    request the second page with the cursor the first response returned.
    """
    from crewai.mcp.client import MCPClient
    from crewai.mcp.transports.stdio import StdioTransport

    client = MCPClient(
        transport=StdioTransport(command=sys.executable, args=[paged_server])
    )
    try:
        await client.connect()
        prompts = await client.list_prompts()
    finally:
        await client.disconnect()

    assert [prompt["name"] for prompt in prompts] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_resolver_discovery_follows_pagination(monkeypatch):
    """Regression guard: HTTPS discovery returned only the first page of tools.

    Without a cursor loop, a tool on the second page is missing from the
    resolved schemas, so an agent can never see or call it.
    """
    import mcp
    import mcp.client.streamable_http

    pages = {
        None: (["page_one_tool"], "cursor-2"),
        "cursor-2": (["page_two_tool"], None),
    }
    requested_cursors = []

    class FakeListToolsResult:
        def __init__(self, names, next_cursor):
            self.tools = [
                type(
                    "Tool",
                    (),
                    {
                        "name": name,
                        "description": f"{name} description",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                )()
                for name in names
            ]
            self.nextCursor = next_cursor

    class FakeSession:
        def __init__(self, read, write, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def list_tools(self, cursor=None, *, params=None):
            requested = cursor if cursor is not None else getattr(params, "cursor", None)
            assert requested in pages, f"server has no page for cursor {requested!r}"
            requested_cursors.append(requested)
            names, next_cursor = pages[requested]
            return FakeListToolsResult(names, next_cursor)

    class FakeStreamableHttpClient:
        def __init__(self, url, *args, **kwargs):
            self.url = url

        async def __aenter__(self):
            return (object(), object(), lambda: "session-id")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        mcp.client.streamable_http, "streamablehttp_client", FakeStreamableHttpClient
    )
    monkeypatch.setattr(mcp, "ClientSession", FakeSession)

    from crewai.mcp.tool_resolver import MCPToolResolver

    resolver = MCPToolResolver(agent=None, logger=logging.getLogger(__name__))
    schemas = await resolver._discover_mcp_tools("https://example.invalid/mcp")

    assert sorted(schemas) == ["page_one_tool", "page_two_tool"]
    assert requested_cursors == [None, "cursor-2"]


@pytest.mark.asyncio
async def test_repeated_prompt_cursor_is_rejected(paged_server):
    """A server that answers every page with the same cursor must not be followed forever.

    Without the guard the loop would request the same page until the discovery
    timeout expires, discarding the prompts it had already collected.
    """
    from crewai.mcp.client import MCPClient
    from crewai.mcp.transports.stdio import StdioTransport

    client = MCPClient(
        transport=StdioTransport(
            command=sys.executable, args=[paged_server, "--repeat-cursor"]
        )
    )
    try:
        await client.connect()
        with pytest.raises(ConnectionError, match="repeated prompts/list pagination cursor"):
            await client.list_prompts()
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_repeated_tool_cursor_is_rejected(monkeypatch):
    """The same guard on the HTTPS discovery path, where no retry wrapper hides it."""
    import mcp
    import mcp.client.streamable_http

    requested_cursors = []

    class FakeListToolsResult:
        def __init__(self):
            self.tools = [
                type(
                    "Tool",
                    (),
                    {
                        "name": "sticky_tool",
                        "description": "sticky",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                )()
            ]
            self.nextCursor = "sticky"

    class FakeSession:
        def __init__(self, read, write, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def list_tools(self, cursor=None, *, params=None):
            requested_cursors.append(
                cursor if cursor is not None else getattr(params, "cursor", None)
            )
            return FakeListToolsResult()

    class FakeStreamableHttpClient:
        def __init__(self, url, *args, **kwargs):
            self.url = url

        async def __aenter__(self):
            return (object(), object(), lambda: "session-id")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(
        mcp.client.streamable_http, "streamablehttp_client", FakeStreamableHttpClient
    )
    monkeypatch.setattr(mcp, "ClientSession", FakeSession)

    from crewai.mcp.tool_resolver import MCPToolResolver

    resolver = MCPToolResolver(agent=None, logger=logging.getLogger(__name__))
    with pytest.raises(RuntimeError, match="repeated tools/list pagination cursor"):
        await resolver._discover_mcp_tools("https://example.invalid/mcp")

    # It gave up on the second page rather than asking for the same one again.
    assert requested_cursors == [None, "sticky"]
