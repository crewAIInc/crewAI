"""Tests for MCPClient's tool-list cache key.

`_mcp_schema_cache` is module-level, so it is shared by every client in the
process. The key therefore has to cover everything that can change what the
server answers -- not only *where* we connect but *as whom*.
"""

import time

import pytest

from crewai.mcp.client import MCPClient, _mcp_schema_cache
from crewai.mcp.transports.http import HTTPTransport
from crewai.mcp.transports.sse import SSETransport
from crewai.mcp.transports.stdio import StdioTransport


@pytest.fixture(autouse=True)
def clear_schema_cache():
    """The cache is module-level state; don't leak entries between tests."""
    _mcp_schema_cache.clear()
    yield
    _mcp_schema_cache.clear()


def _key(transport) -> str:
    return MCPClient(transport, cache_tools_list=True)._get_cache_key("tools")


def test_http_clients_with_different_auth_headers_do_not_share_a_cache_entry():
    """Two tenants on the same URL must not see each other's tool list.

    `headers={"Authorization": "Bearer ..."}` alongside `cache_tools_list=True` is
    the documented usage for a remote MCP server, and an MCP server routinely
    returns a different tool set per caller.
    """
    tenant_a = _key(
        HTTPTransport(
            url="https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer TENANT-A"},
        )
    )
    tenant_b = _key(
        HTTPTransport(
            url="https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer TENANT-B"},
        )
    )

    assert tenant_a != tenant_b

    # And concretely: what A caches must not be served to B.
    _mcp_schema_cache[tenant_a] = ([{"name": "tenant_a_only_tool"}], time.time())
    assert tenant_b not in _mcp_schema_cache


def test_stdio_clients_with_different_env_do_not_share_a_cache_entry():
    """Same command, different credentials in `env` -- also documented usage."""
    account_1 = _key(
        StdioTransport(
            command="python", args=["server.py"], env={"API_KEY": "key-1"}
        )
    )
    account_2 = _key(
        StdioTransport(
            command="python", args=["server.py"], env={"API_KEY": "key-2"}
        )
    )

    assert account_1 != account_2


def test_sse_clients_with_different_auth_headers_do_not_share_a_cache_entry():
    """SSE takes `headers` too, and had the same defect."""
    assert _key(
        SSETransport(url="https://s/sse", headers={"Authorization": "Bearer A"})
    ) != _key(SSETransport(url="https://s/sse", headers={"Authorization": "Bearer B"}))


def test_streamable_flag_changes_the_cache_key():
    """`streamable` picks a different protocol, so it may yield a different list."""
    assert _key(HTTPTransport(url="https://s/mcp", streamable=True)) != _key(
        HTTPTransport(url="https://s/mcp", streamable=False)
    )


def test_identical_transports_still_share_a_cache_entry():
    """Control: the cache must still hit for genuinely equivalent clients.

    Header order and dict insertion order are not meaningful, so they must not
    split the key -- otherwise the fix would quietly disable caching.
    """
    first = _key(
        HTTPTransport(
            url="https://mcp.example.com/mcp",
            headers={"Authorization": "Bearer T", "X-Trace": "1"},
        )
    )
    second = _key(
        HTTPTransport(
            url="https://mcp.example.com/mcp",
            headers={"X-Trace": "1", "Authorization": "Bearer T"},
        )
    )

    assert first == second


def test_credentials_are_not_present_verbatim_in_the_cache_key():
    """Keys get logged and shown in errors, so hash the credential-bearing parts."""
    secret = "Bearer super-secret-token"
    key = _key(HTTPTransport(url="https://s/mcp", headers={"Authorization": secret}))

    assert secret not in key
    assert "super-secret-token" not in key


def test_resource_type_still_separates_entries():
    """Control: the existing per-resource-type separation must be preserved."""
    client = MCPClient(HTTPTransport(url="https://s/mcp"), cache_tools_list=True)

    assert client._get_cache_key("tools") != client._get_cache_key("prompts")


def test_no_credentials_still_yields_a_stable_key():
    """Control: the common case -- no headers at all -- must be deterministic."""
    assert _key(HTTPTransport(url="https://s/mcp")) == _key(
        HTTPTransport(url="https://s/mcp")
    )
