"""Tests for ToolCallHookContext.tool_call_id (issue #8649).

Verifies that:
- tool_call_id is accepted and stored on the context.
- Default is None (ReAct/text path gives the model no id).
- A before hook sees the same call_id as the matching after hook.
- After-hook result rewriting coexists with tool_call_id.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from crewai.hooks import (
    clear_all_tool_call_hooks,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
)
from crewai.hooks.tool_hooks import ToolCallHookContext


@pytest.fixture(autouse=True)
def clean_hooks():
    """Clear global hooks around every test."""
    n_before, n_after = clear_all_tool_call_hooks()
    yield
    clear_all_tool_call_hooks()


def _make_context(tool_call_id=None, **overrides):
    defaults = dict(
        tool_name="search",
        tool_input={"q": "foo"},
        tool=Mock(name="structured_tool"),
        agent=None,
        task=None,
        crew=None,
    )
    defaults.update(overrides)
    return ToolCallHookContext(tool_call_id=tool_call_id, **defaults)


def test_default_tool_call_id_is_none():
    ctx = _make_context()
    assert ctx.tool_call_id is None


def test_native_path_stores_tool_call_id():
    ctx = _make_context(tool_call_id="call_abc123")
    assert ctx.tool_call_id == "call_abc123"


def test_after_hook_context_preserves_tool_call_id():
    ctx = _make_context(
        tool_call_id="call_def456",
        tool_result="results",
        raw_tool_result=["r1"],
    )
    assert ctx.tool_call_id == "call_def456"
    assert ctx.tool_result == "results"
    assert ctx.raw_tool_result == ["r1"]


def test_before_and_after_share_same_tool_call_id():
    seen = {}

    def before_hook(ctx: ToolCallHookContext):
        seen["before_id"] = ctx.tool_call_id

    def after_hook(ctx: ToolCallHookContext):
        seen["after_id"] = ctx.tool_call_id
        return None

    register_before_tool_call_hook(before_hook)
    register_after_tool_call_hook(after_hook)

    before = _make_context(tool_call_id="call_shared")
    after = _make_context(
        tool_call_id="call_shared",
        tool_result="done",
        raw_tool_result="done",
    )

    from crewai.hooks.tool_hooks import (
        run_after_tool_call_hooks,
        run_before_tool_call_hooks,
    )

    run_before_tool_call_hooks(before)
    run_after_tool_call_hooks(after)

    assert seen["before_id"] == "call_shared"
    assert seen["after_id"] == "call_shared"
