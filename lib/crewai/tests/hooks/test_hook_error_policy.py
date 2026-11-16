"""A hook can opt into failing closed when it raises (#9434)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from crewai.agents.parser import AgentAction
from crewai.hooks import (
    HookAborted,
    clear_all_tool_call_hooks,
    set_error_policy,
    unregister_before_tool_call_hook,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    run_after_tool_call_hooks,
    run_before_tool_call_hooks,
)
from crewai.tools import tool
from crewai.utilities.i18n import I18N
from crewai.utilities.tool_utils import execute_tool_and_check_finality


@pytest.fixture(autouse=True)
def _clear_hooks():
    clear_all_tool_call_hooks()
    yield
    clear_all_tool_call_hooks()


def _context() -> ToolCallHookContext:
    mock_tool = Mock()
    mock_tool.name = "send_email"
    return ToolCallHookContext(
        tool_name="send_email",
        tool_input={"to": "x@example.com"},
        tool=mock_tool,
        tool_result="sent",
    )


def _failing_hook(context: ToolCallHookContext) -> None:
    raise RuntimeError("policy service unreachable")


def test_a_failing_before_hook_still_fails_open_by_default() -> None:
    register_before_tool_call_hook(_failing_hook)

    assert run_before_tool_call_hooks(_context()) is False


def test_a_before_hook_can_opt_into_failing_closed() -> None:
    register_before_tool_call_hook(_failing_hook, on_error="abort")

    assert run_before_tool_call_hooks(_context()) is True


def test_an_after_hook_can_opt_into_failing_closed() -> None:
    register_after_tool_call_hook(_failing_hook, on_error="abort")

    with pytest.raises(HookAborted, match="policy service unreachable"):
        run_after_tool_call_hooks(_context())


def test_a_hook_aborted_reason_is_kept() -> None:
    def policy(context: ToolCallHookContext) -> None:
        raise HookAborted("blocked by policy", source="policy")

    register_after_tool_call_hook(policy, on_error="abort")

    with pytest.raises(HookAborted, match="blocked by policy"):
        run_after_tool_call_hooks(_context())


def test_unregistering_the_hook_drops_its_policy() -> None:
    register_before_tool_call_hook(_failing_hook, on_error="abort")
    assert unregister_before_tool_call_hook(_failing_hook) is True

    register_before_tool_call_hook(_failing_hook)

    assert run_before_tool_call_hooks(_context()) is False


def test_a_failing_policy_hook_blocks_the_tool_call() -> None:
    """The reproduction from the issue: the tool body must not run."""
    ran: list[str] = []

    @tool("send_email")
    def send_email(to: str) -> str:
        """Send an email."""
        ran.append(to)
        return "sent"

    def policy(context: ToolCallHookContext) -> None:
        raise RuntimeError("policy service unreachable")

    register_before_tool_call_hook(policy, on_error="abort")

    action = AgentAction(
        thought="",
        tool="send_email",
        tool_input='{"to": "x@example.com"}',
        text='Action: send_email\nAction Input: {"to": "x@example.com"}',
    )
    tool_result = execute_tool_and_check_finality(
        action, [send_email.to_structured_tool()], I18N()
    )

    assert ran == []
    assert "blocked by hook" in tool_result.result


def test_a_filtered_hook_keeps_its_policy():
    """A filtered hook is registered as a wrapper; the policy must still apply."""
    from crewai.hooks import before_tool_call

    @before_tool_call(tools=["send_email"])
    def policy(context: ToolCallHookContext) -> None:
        raise RuntimeError("policy service unreachable")

    set_error_policy(policy, "abort")

    assert run_before_tool_call_hooks(_context()) is True


def test_a_filtered_hook_that_does_not_match_stays_out_of_the_way():
    """The wrapper skips other tools before the policy can matter."""
    from crewai.hooks import before_tool_call

    @before_tool_call(tools=["other_tool"])
    def policy(context: ToolCallHookContext) -> None:
        raise RuntimeError("policy service unreachable")

    set_error_policy(policy, "abort")

    assert run_before_tool_call_hooks(_context()) is False


def test_an_on_hook_keeps_its_policy():
    """``@on`` stores the wrapper on the original as ``_registered_hook``."""
    from crewai.hooks import InterceptionPoint, on

    @on(InterceptionPoint.PRE_TOOL_CALL, tools=["send_email"])
    def policy(context: ToolCallHookContext) -> None:
        raise RuntimeError("policy service unreachable")

    set_error_policy(policy, "abort")

    assert run_before_tool_call_hooks(_context()) is True


def test_unregistering_a_filtered_hook_drops_its_policy():
    from crewai.hooks import before_tool_call, clear_all_tool_call_hooks

    @before_tool_call(tools=["send_email"])
    def policy(context: ToolCallHookContext) -> None:
        raise RuntimeError("policy service unreachable")

    set_error_policy(policy, "abort")
    clear_all_tool_call_hooks()

    register_before_tool_call_hook(policy)

    assert run_before_tool_call_hooks(_context()) is False


class _HookOwner:
    """Stands in for a ``@CrewBase`` class: the decorator returns the plain function."""

    def guard(self, context: ToolCallHookContext) -> None:
        raise RuntimeError("policy service unreachable")


def test_a_crew_method_keeps_a_policy_set_on_the_class_function():
    """A ``@CrewBase`` method is registered as a bound method, the caller holds the
    plain function, so the policy must resolve through ``__func__``."""
    owner = _HookOwner()
    register_before_tool_call_hook(owner.guard)
    set_error_policy(_HookOwner.guard, "abort")

    assert run_before_tool_call_hooks(_context()) is True


def test_a_crew_method_keeps_a_policy_set_on_the_bound_method():
    owner = _HookOwner()
    register_before_tool_call_hook(owner.guard)
    set_error_policy(owner.guard, "abort")

    assert run_before_tool_call_hooks(_context()) is True


def test_a_filtered_crew_method_keeps_its_policy():
    """The crew-base filter closures must stay linked to the method they wrap."""
    from crewai.project.crew_base import _register_crew_hooks

    class Crew:
        def guard(self, context: ToolCallHookContext) -> None:
            raise RuntimeError("policy service unreachable")

    Crew.guard.is_before_tool_call_hook = True  # type: ignore[attr-defined]
    Crew.guard._filter_tools = ["send_email"]  # type: ignore[attr-defined]
    set_error_policy(Crew.guard, "abort")

    _register_crew_hooks(Crew(), Crew)

    assert run_before_tool_call_hooks(_context()) is True
