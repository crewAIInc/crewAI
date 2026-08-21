"""Compatibility helpers for the MCP Python SDK 1.x and 2.x.

mcp 2.0 renamed ``streamablehttp_client`` → ``streamable_http_client``,
changed its yield from a 3-tuple to a 2-tuple, dropped the ``headers``
kwarg (headers go on an ``httpx`` client via ``create_mcp_http_client``),
and switched camelCase field access (``inputSchema``, ``isError``) to
snake_case (``input_schema``, ``is_error``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


def unpack_transport_streams(streams: tuple[Any, ...]) -> tuple[Any, Any]:
    """Normalize streamable-HTTP context yields across mcp 1.x and 2.x.

    1.x yields ``(read, write, get_session_id)``; 2.x yields ``(read, write)``.
    """
    if len(streams) == 3:
        read, write, _ = streams
        return read, write
    if len(streams) == 2:
        read, write = streams
        return read, write
    raise ValueError(
        f"Unexpected streamable HTTP transport yield with {len(streams)} values"
    )


def create_streamable_http_client(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    terminate_on_close: bool = True,
    http_client: Any | None = None,
) -> Any:
    """Build a streamable-HTTP async context manager for mcp 1.x and 2.x.

    Prefer ``streamable_http_client`` (present in both lines). When *headers*
    are needed, pass a pre-built *http_client* from
    :func:`create_mcp_http_client`, or let this helper construct one. The
    caller must enter the returned context manager (and the http client when
    it owns it — see :func:`open_streamable_http`).
    """
    from mcp.client.streamable_http import streamable_http_client

    if http_client is None and headers:
        from mcp.shared._httpx_utils import create_mcp_http_client

        http_client = create_mcp_http_client(headers=headers)

    if http_client is not None:
        return streamable_http_client(
            url,
            http_client=http_client,
            terminate_on_close=terminate_on_close,
        )
    return streamable_http_client(url, terminate_on_close=terminate_on_close)


@asynccontextmanager
async def open_streamable_http(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    terminate_on_close: bool = True,
) -> AsyncIterator[tuple[Any, Any]]:
    """Open streamable HTTP transport; yield ``(read, write)`` on 1.x and 2.x."""
    from contextlib import AsyncExitStack

    from mcp.shared._httpx_utils import create_mcp_http_client

    async with AsyncExitStack() as stack:
        http_client = None
        if headers:
            http_client = create_mcp_http_client(headers=headers)
            await stack.enter_async_context(http_client)

        transport_cm = create_streamable_http_client(
            url,
            terminate_on_close=terminate_on_close,
            http_client=http_client,
        )
        streams = await stack.enter_async_context(transport_cm)
        yield unpack_transport_streams(streams)


def tool_input_schema(tool: Any) -> dict[str, Any]:
    """Return a tool input schema across mcp 1.x (``inputSchema``) and 2.x."""
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema or {}


def call_tool_is_error(result: Any) -> bool:
    """Return CallToolResult error flag across mcp 1.x (``isError``) and 2.x."""
    value = getattr(result, "is_error", None)
    if value is None:
        value = getattr(result, "isError", None)
    return bool(value)
