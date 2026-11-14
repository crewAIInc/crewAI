from crewai.mcp.filters import create_static_tool_filter


def test_static_filter_matches_original_mcp_tool_name() -> None:
    tool_filter = create_static_tool_filter(allowed_tool_names=["delete-file"])

    assert tool_filter({"name": "delete_file", "original_name": "delete-file"})


def test_static_filter_matches_sanitized_configured_name() -> None:
    tool_filter = create_static_tool_filter(allowed_tool_names=["delete_file"])

    assert tool_filter({"name": "delete_file", "original_name": "delete-file"})


def test_static_filter_blocklist_matches_original_mcp_tool_name() -> None:
    tool_filter = create_static_tool_filter(blocked_tool_names=["delete-file"])

    assert not tool_filter({"name": "delete_file", "original_name": "delete-file"})
