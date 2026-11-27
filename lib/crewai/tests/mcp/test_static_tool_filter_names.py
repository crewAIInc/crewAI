"""Static tool filters must match published MCP tool names too."""

from crewai.mcp.filters import create_static_tool_filter


def test_blocked_name_matches_original_name():
    listed = {"name": "delete_file", "original_name": "delete-file"}

    assert create_static_tool_filter(blocked_tool_names=["delete-file"])(listed) is False


def test_allowed_name_matches_original_name():
    listed = {"name": "you_search", "original_name": "you-search"}

    assert create_static_tool_filter(allowed_tool_names=["you-search"])(listed) is True


def test_blocklist_survives_sanitizer_change():
    # A configured name in the old sanitized form still matches even if the
    # sanitizer now prefixes digit-leading names with an underscore.
    listed = {"name": "_3d_renderer", "original_name": "3d_renderer"}

    assert create_static_tool_filter(blocked_tool_names=["3d_renderer"])(listed) is False
