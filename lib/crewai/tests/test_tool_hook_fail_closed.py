"""Tests for Crew(tool_hook_fail_closed) and ToolCallHookContext.tool_call_id.

Covers crewAIInc/crewAI #9434 (tool hook exceptions fail open by default)
and #8649 (ToolCallHookContext has no tool call id).

NOTE: the test path is proposed; adjust it to the repo's test layout at PR
time. The crew is stubbed with SimpleNamespace: the runners read the flag
via getattr(crew, "tool_hook_fail_closed", False), so a stub exercises the
same path as a real Crew; one test below also constructs a real Crew to
assert the field default.
"""

from types import SimpleNamespace

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.hooks.dispatch import InterceptionPoint, clear
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    run_after_tool_call_hooks,
    run_before_tool_call_hooks,
)


def _context(*, crew=None, tool_call_id=None, tool_result=None):
    return ToolCallHookContext(
        tool_name="search",
        tool_input={"q": "x"},
        tool=SimpleNamespace(name="search"),  # type: ignore[arg-type]
        crew=crew,
        tool_call_id=tool_call_id,
        tool_result=tool_result,
    )


@pytest.fixture(autouse=True)
def _clear_hooks():
    yield
    clear(InterceptionPoint.PRE_TOOL_CALL)
    clear(InterceptionPoint.POST_TOOL_CALL)


def _crew(**kwargs):
    # Current main's Crew validator requires agents/tasks/config; a bare
    # Crew() no longer validates, so build the lightest valid crew.
    return Crew(agents=[Agent(role="r", goal="g", backstory="b")], tasks=[], **kwargs)


def test_crew_field_defaults_to_fail_open():
    assert _crew().tool_hook_fail_closed is False
    assert _crew(tool_hook_fail_closed=True).tool_hook_fail_closed is True


def test_before_hook_error_fail_open_by_default():
    def boom(context):
        raise RuntimeError("boom")

    register_before_tool_call_hook(boom)
    assert run_before_tool_call_hooks(_context()) is False


def test_before_hook_error_fail_closed_blocks():
    def boom(context):
        raise RuntimeError("boom")

    register_before_tool_call_hook(boom)
    crew = SimpleNamespace(tool_hook_fail_closed=True)
    assert run_before_tool_call_hooks(_context(crew=crew)) is True


def test_before_hook_error_fail_closed_explicit_on_error():
    def boom(context):
        raise RuntimeError("boom")

    register_before_tool_call_hook(boom)
    assert run_before_tool_call_hooks(_context(), on_error="abort") is True


def test_after_hook_error_fail_open_by_default():
    def boom(context):
        raise RuntimeError("boom")

    register_after_tool_call_hook(boom)
    assert run_after_tool_call_hooks(_context(tool_result="real result")) == "real result"


def test_after_hook_error_fail_closed_withholds_result():
    def boom(context):
        raise RuntimeError("boom")

    register_after_tool_call_hook(boom)
    crew = SimpleNamespace(tool_hook_fail_closed=True)
    result = run_after_tool_call_hooks(_context(crew=crew, tool_result="real result"))
    assert result is not None
    assert result.startswith("Tool result withheld")
    assert "real result" not in result


def test_tool_call_id_visible_to_hook():
    seen = []

    def record(context):
        seen.append(context.tool_call_id)

    register_before_tool_call_hook(record)
    run_before_tool_call_hooks(_context(tool_call_id="call_abc"))
    assert seen == ["call_abc"]


def test_tool_call_id_none_on_react_path():
    seen = []

    def record(context):
        seen.append(context.tool_call_id)

    register_before_tool_call_hook(record)
    run_before_tool_call_hooks(_context(tool_call_id=None))
    assert seen == [None]
