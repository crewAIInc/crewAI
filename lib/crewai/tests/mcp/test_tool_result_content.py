"""MCP tool results must reach the agent with every content block."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as mcp_types
import pytest

from crewai.mcp.client import MCPClient
from crewai.mcp.content import content_blocks_to_text
from crewai.tools.mcp_native_tool import MCPNativeTool
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


THREE_TEXT_BLOCKS = mcp_types.CallToolResult(
    content=[
        mcp_types.TextContent(type="text", text=text)
        for text in ("first", "second", "third")
    ]
)

MIXED_BLOCKS = mcp_types.CallToolResult(
    content=[
        mcp_types.TextContent(type="text", text="chart below"),
        mcp_types.ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png"),
        mcp_types.EmbeddedResource(
            type="resource",
            resource=mcp_types.BlobResourceContents(
                uri="file:///report.pdf", blob="JVBERi0=", mimeType="application/pdf"
            ),
        ),
    ]
)


def _via_client(result: mcp_types.CallToolResult) -> str:
    """MCPClient, which feeds MCPNativeTool on the MCPServerConfig path."""
    client = MCPClient(MagicMock())
    client._session = MagicMock(call_tool=AsyncMock(return_value=result))
    return asyncio.run(client._call_tool_impl("search", {})).content


def _via_tool_wrapper(result: mcp_types.CallToolResult) -> Any:
    """MCPToolWrapper, used for plain https:// MCP server URLs."""

    @asynccontextmanager
    async def fake_streamablehttp_client(*_args: Any, **_kwargs: Any) -> Any:
        yield (None, None, None)

    session = MagicMock(
        initialize=AsyncMock(), call_tool=AsyncMock(return_value=result)
    )
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
    ):
        return tool.run()


CALL_PATHS = pytest.mark.parametrize(
    "call_tool",
    [_via_client, _via_tool_wrapper, MCPNativeTool._extract_content],
    ids=["mcp_client", "tool_wrapper", "native_tool_extract"],
)


@CALL_PATHS
def test_every_text_block_reaches_the_agent(call_tool: Any) -> None:
    """All text blocks are kept, in order, not just the first one."""
    assert call_tool(THREE_TEXT_BLOCKS) == "first\nsecond\nthird"


@CALL_PATHS
def test_non_text_blocks_become_placeholders(call_tool: Any) -> None:
    """Image and blob blocks become placeholders without their base64 data."""
    assert call_tool(MIXED_BLOCKS) == (
        "chart below\n[image: image/png]\n[resource: file:///report.pdf]"
    )


def test_remaining_block_types() -> None:
    """Audio and resource links get placeholders; embedded text is kept."""
    blocks = [
        mcp_types.AudioContent(type="audio", data="UklGRg==", mimeType="audio/wav"),
        mcp_types.ResourceLink(
            type="resource_link", uri="file:///notes.md", name="notes"
        ),
        mcp_types.EmbeddedResource(
            type="resource",
            resource=mcp_types.TextResourceContents(
                uri="file:///inline.txt", text="inline text"
            ),
        ),
    ]
    assert content_blocks_to_text(blocks) == (
        "[audio: audio/wav]\n[resource_link: file:///notes.md]\ninline text"
    )
