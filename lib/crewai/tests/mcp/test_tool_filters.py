"""Tests for static MCP tool filters.

MCPClient lists tools as ``{"name": sanitize_tool_name(tool.name),
"original_name": tool.name, ...}`` and the static filter receives those
dicts, while users write allow/block lists with the tool names the MCP
server publishes (for example ``you-search`` or ``deleteFile``).
"""

import pytest
from crewai.mcp.filters import create_static_tool_filter
from crewai.utilities.string_utils import sanitize_tool_name


def listed_tool(server_name: str) -> dict[str, str]:
    """Build a tool dict the way MCPClient lists it."""
    return {"name": sanitize_tool_name(server_name), "original_name": server_name}


@pytest.mark.parametrize("server_name", ["delete-file", "deleteFile", "Delete File"])
def test_blocked_server_name_blocks_listed_tool(server_name: str) -> None:
    tool_filter = create_static_tool_filter(blocked_tool_names=[server_name])

    assert tool_filter(listed_tool(server_name)) is False


def test_allowed_server_names_allow_listed_tools() -> None:
    tool_filter = create_static_tool_filter(
        allowed_tool_names=["you-search", "you-research"]
    )

    assert tool_filter(listed_tool("you-search")) is True
    assert tool_filter(listed_tool("you-research")) is True
    assert tool_filter(listed_tool("you-contents")) is False


def test_normalized_names_keep_working() -> None:
    tool_filter = create_static_tool_filter(
        allowed_tool_names=["read_file", "delete_file"],
        blocked_tool_names=["delete_file"],
    )

    assert tool_filter(listed_tool("read_file")) is True
    assert tool_filter(listed_tool("delete_file")) is False
    assert tool_filter(listed_tool("write_file")) is False


def test_block_takes_precedence_across_spellings() -> None:
    tool_filter = create_static_tool_filter(
        allowed_tool_names=["delete_file"],
        blocked_tool_names=["delete-file"],
    )

    assert tool_filter(listed_tool("delete-file")) is False
