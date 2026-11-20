from unittest.mock import MagicMock

import pytest

from crewai.hooks import (
    HookAborted,
    InterceptionPoint,
    after_tool_call,
    before_tool_call,
    clear_all_global_hooks,
    clear_all_hooks,
    on,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    unregister_after_tool_call_hook,
    unregister_before_tool_call_hook,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    run_after_tool_call_hooks,
    run_before_tool_call_hooks,
)


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_all_global_hooks()
    clear_all_hooks()
    yield
    clear_all_global_hooks()
    clear_all_hooks()


def _make_context(tool_result: str | None = None) -> ToolCallHookContext:
    return ToolCallHookContext(
        tool_name="sensitive_tool",
        tool_input={"query": "SELECT * FROM users"},
        tool=MagicMock(),
        tool_result=tool_result,
    )


def test_before_tool_call_defaults_to_fail_open():
    def broken_guard(ctx: ToolCallHookContext) -> None:
        raise RuntimeError("guardrail crashed")

    register_before_tool_call_hook(broken_guard)

    assert run_before_tool_call_hooks(_make_context()) is False


def test_before_tool_call_fail_closed_blocks_on_exception():
    def broken_guard(ctx: ToolCallHookContext) -> None:
        raise RuntimeError("OPA unreachable")

    register_before_tool_call_hook(broken_guard, on_error="abort")

    assert run_before_tool_call_hooks(_make_context()) is True
    assert unregister_before_tool_call_hook(broken_guard) is True


def test_after_tool_call_fail_closed_raises_hook_aborted():
    original = ValueError("redaction regex failure")

    def broken_redactor(ctx: ToolCallHookContext) -> str | None:
        raise original

    register_after_tool_call_hook(broken_redactor, on_error="abort")

    with pytest.raises(HookAborted) as exc_info:
        run_after_tool_call_hooks(_make_context(tool_result="SECRET_TOKEN=123"))

    assert "redaction regex failure" in exc_info.value.reason
    assert exc_info.value.__cause__ is original
    assert unregister_after_tool_call_hook(broken_redactor) is True


def test_fail_closed_decorators_and_generic_on():
    @before_tool_call(tools=["sensitive_tool"], on_error="abort")
    def decorator_before(ctx: ToolCallHookContext) -> None:
        raise KeyError("missing policy key")

    assert run_before_tool_call_hooks(_make_context()) is True
    clear_all_global_hooks()

    @after_tool_call(on_error="abort")
    def decorator_after(ctx: ToolCallHookContext) -> str | None:
        raise RuntimeError("post-hook failure")

    with pytest.raises(HookAborted):
        run_after_tool_call_hooks(_make_context(tool_result="raw"))
    clear_all_global_hooks()

    @on(InterceptionPoint.PRE_TOOL_CALL, on_error="abort")
    def generic_on_hook(ctx: ToolCallHookContext) -> None:
        raise RuntimeError("generic interception failure")

    assert run_before_tool_call_hooks(_make_context()) is True


def test_invalid_on_error_policy_raises_value_error():
    def noop(ctx: ToolCallHookContext) -> None:
        return None

    with pytest.raises(ValueError, match="Invalid on_error policy"):
        register_before_tool_call_hook(noop, on_error="invalid")  # type: ignore[arg-type]