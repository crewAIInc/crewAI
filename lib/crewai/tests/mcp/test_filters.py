"""Tests for MCP tool filters."""

from crewai.mcp.filters import create_static_tool_filter


def test_static_filter_matches_original_mcp_name() -> None:
    """Published names should match MCP definitions after sanitization."""
    filter_fn = create_static_tool_filter(allowed_tool_names=["delete-file"])

    assert filter_fn({"name": "delete_file", "original_name": "delete-file"})


def test_static_filter_matches_configured_camel_case_name() -> None:
    """Configured names should work whether they use MCP or sanitized casing."""
    filter_fn = create_static_tool_filter(allowed_tool_names=["deleteFile"])

    assert filter_fn({"name": "delete_file", "original_name": "deleteFile"})


def test_blocked_original_name_takes_precedence() -> None:
    """A blocked tool remains blocked even when it is also allowed."""
    filter_fn = create_static_tool_filter(
        allowed_tool_names=["delete_file"],
        blocked_tool_names=["delete-file"],
    )

    assert not filter_fn({"name": "delete_file", "original_name": "delete-file"})


def test_static_filter_supports_definitions_without_original_name() -> None:
    """Definitions from older callers that only contain ``name`` still work."""
    filter_fn = create_static_tool_filter(allowed_tool_names=["read-file"])

    assert filter_fn({"name": "read_file"})
