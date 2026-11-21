"""MCPToolWrapper must not report non-timeout MCP errors as timeouts."""

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


class HTTPStatusError(Exception):
    """Stand-in for an HTTP 401/404 raised by the MCP transport."""


def _run_with_error(error: BaseException) -> tuple[str, AsyncMock]:
    @asynccontextmanager
    async def fake_streamablehttp_client(*_args: Any, **_kwargs: Any) -> Any:
        yield (None, None, None)

    call_tool = AsyncMock(side_effect=error)
    session = MagicMock(initialize=AsyncMock(), call_tool=call_tool)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    tool = MCPToolWrapper(
        mcp_server_params={"url": "https://mcp.example.test/mcp"},
        tool_name="search",
        tool_schema={"description": "search"},
        server_name="demo",
    )
    with (
        patch(
            "mcp.client.streamable_http.streamablehttp_client",
            fake_streamablehttp_client,
        ),
        patch("mcp.ClientSession", return_value=session),
        patch("crewai.tools.mcp_tool_wrapper.asyncio.sleep", AsyncMock()),
    ):
        return tool.run(), call_tool


def test_group_with_http_error_is_reported_as_that_error_and_not_retried() -> None:
    group = BaseExceptionGroup(
        "unhandled errors in a TaskGroup",
        [HTTPStatusError("Client error '401 Unauthorized' for url 'https://x.test'")],
    )
    message, call_tool = _run_with_error(group)

    assert "HTTPStatusError" in message
    assert "401" in message
    assert "timed out" not in message.lower()
    assert call_tool.await_count == 1


def test_group_with_timeout_is_still_reported_as_timeout() -> None:
    group = BaseExceptionGroup(
        "unhandled errors in a TaskGroup",
        [asyncio.TimeoutError("read timed out")],
    )
    message, call_tool = _run_with_error(group)

    assert "timed out" in message.lower()
    assert call_tool.await_count == 3
