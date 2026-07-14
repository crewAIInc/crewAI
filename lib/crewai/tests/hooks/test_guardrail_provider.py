"""Focused contract tests for the GuardrailProvider adapter."""

from __future__ import annotations

import pytest

from crewai.hooks import (
    GuardrailDecision,
    GuardrailRequest,
    clear_before_tool_call_hooks,
    enable_guardrail,
    get_before_tool_call_hooks,
    register_before_tool_call_hook,
    unregister_before_tool_call_hook,
)


class AllowProvider:
    """Minimal provider fixture that always allows the tool call."""

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return GuardrailDecision(allow=True)


@pytest.fixture(autouse=True)
def restore_before_tool_hooks():
    """Keep the global hook registry unchanged outside this test module."""

    original_hooks = get_before_tool_call_hooks()
    clear_before_tool_call_hooks()
    yield
    clear_before_tool_call_hooks()
    for hook in original_hooks:
        register_before_tool_call_hook(hook)


def test_enable_guardrail_returns_unregisterable_hook():
    """The adapter hook can be removed through the existing registry API."""

    hook = enable_guardrail(AllowProvider())

    assert get_before_tool_call_hooks() == [hook]
    assert unregister_before_tool_call_hook(hook) is True
    assert get_before_tool_call_hooks() == []
