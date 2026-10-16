from crewai.mcp.filters import (
    StaticToolFilter,
    create_dynamic_tool_filter,
    create_static_tool_filter,
)


def test_static_tool_filter_none_allowed_permits_all():
    """allowed_tool_names=None should allow all tools."""
    filter_fn = StaticToolFilter(allowed_tool_names=None)
    assert filter_fn({"name": "any_tool"}) is True
    assert filter_fn({"name": "another_tool"}) is True


def test_static_tool_filter_empty_list_blocks_all():
    """allowed_tool_names=[] should block all tools (Issue #7778)."""
    filter_fn = StaticToolFilter(allowed_tool_names=[])
    assert filter_fn({"name": "read_file"}) is False
    assert filter_fn({"name": "write_file"}) is False


def test_static_tool_filter_explicit_allowlist():
    """Only tools in allowed_tool_names should be permitted."""
    filter_fn = StaticToolFilter(allowed_tool_names=["read_file", "search"])
    assert filter_fn({"name": "read_file"}) is True
    assert filter_fn({"name": "search"}) is True
    assert filter_fn({"name": "write_file"}) is False


def test_static_tool_filter_blocked_precedence():
    """Blocked tools take precedence over allowed tools."""
    filter_fn = StaticToolFilter(
        allowed_tool_names=["read_file", "delete_file"],
        blocked_tool_names=["delete_file"],
    )
    assert filter_fn({"name": "read_file"}) is True
    assert filter_fn({"name": "delete_file"}) is False


def test_create_static_tool_filter_factory():
    """Factory create_static_tool_filter properly constructs StaticToolFilter."""
    filter_fn = create_static_tool_filter(allowed_tool_names=[])
    assert filter_fn({"name": "any_tool"}) is False

    filter_fn_none = create_static_tool_filter(allowed_tool_names=None)
    assert filter_fn_none({"name": "any_tool"}) is True
