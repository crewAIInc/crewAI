"""Tests for fail-closed behavior of unsafe tools.

Unsafe tools (marked with unsafe=True) must be denied by default unless
a before_tool_call hook explicitly returns True. This enforces a fail-closed
security model where high-impact tools cannot run without an explicit safety
policy.

See: https://github.com/crewAIInc/crewAI/issues/4593
"""

from __future__ import annotations

import pytest

from crewai.hooks import tool_hooks
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    register_before_tool_call_hook,
)
from crewai.tools import BaseTool, tool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.utilities.tool_utils import execute_tool_and_check_finality


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    original_before = tool_hooks._before_tool_call_hooks.copy()
    original_after = tool_hooks._after_tool_call_hooks.copy()

    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()

    yield

    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.extend(original_before)
    tool_hooks._after_tool_call_hooks.extend(original_after)


class TestUnsafeFlagOnBaseTool:
    """Test that the unsafe flag can be set on BaseTool subclasses."""

    def test_default_unsafe_is_false(self):
        """Tools are safe by default (unsafe=False)."""
        from pydantic import BaseModel

        class MyTool(BaseTool):
            name: str = "my_tool"
            description: str = "A test tool"

            def _run(self, **kwargs) -> str:
                return "result"

        t = MyTool()
        assert t.unsafe is False

    def test_unsafe_can_be_set_true(self):
        """Tools can be marked as unsafe."""
        from pydantic import BaseModel

        class UnsafeTool(BaseTool):
            name: str = "unsafe_tool"
            description: str = "A dangerous tool"
            unsafe: bool = True

            def _run(self, **kwargs) -> str:
                return "dangerous result"

        t = UnsafeTool()
        assert t.unsafe is True


class TestUnsafeFlagOnToolDecorator:
    """Test that the @tool decorator supports unsafe=True."""

    def test_tool_decorator_default_safe(self):
        """@tool decorator creates safe tools by default."""

        @tool("safe_tool")
        def safe_tool(x: str) -> str:
            """A safe tool."""
            return x

        assert safe_tool.unsafe is False

    def test_tool_decorator_unsafe_true(self):
        """@tool decorator supports unsafe=True."""

        @tool("unsafe_tool", unsafe=True)
        def unsafe_tool(x: str) -> str:
            """An unsafe tool."""
            return x

        assert unsafe_tool.unsafe is True

    def test_tool_decorator_unsafe_propagates_to_structured_tool(self):
        """unsafe flag propagates when converting to CrewStructuredTool."""

        @tool("unsafe_tool", unsafe=True)
        def unsafe_tool(x: str) -> str:
            """An unsafe tool."""
            return x

        structured = unsafe_tool.to_structured_tool()
        assert structured.unsafe is True


class TestUnsafeFlagOnCrewStructuredTool:
    """Test that CrewStructuredTool stores the unsafe flag."""

    def test_default_unsafe_is_false(self):
        """CrewStructuredTool defaults to unsafe=False."""
        from pydantic import BaseModel

        class Schema(BaseModel):
            x: str

        t = CrewStructuredTool(
            name="test",
            description="test",
            args_schema=Schema,
            func=lambda x: x,
        )
        assert t.unsafe is False

    def test_unsafe_can_be_set_true(self):
        """CrewStructuredTool accepts unsafe=True."""
        from pydantic import BaseModel

        class Schema(BaseModel):
            x: str

        t = CrewStructuredTool(
            name="test",
            description="test",
            args_schema=Schema,
            func=lambda x: x,
            unsafe=True,
        )
        assert t.unsafe is True


class TestFailClosedBehavior:
    """Test fail-closed enforcement for unsafe tools.

    These tests verify that:
    - Unsafe tools are blocked when no hooks are registered
    - Unsafe tools are blocked when hooks return None (neutral)
    - Unsafe tools execute when a hook explicitly returns True
    - Safe tools (default) are unaffected by the fail-closed logic
    - Hook returning False still blocks safe tools (existing behavior preserved)

    Example usage for users:

        from crewai.tools import tool
        from crewai.hooks.tool_hooks import register_before_tool_call_hook, ToolCallHookContext

        # 1. Mark a tool as unsafe
        @tool("delete_database", unsafe=True)
        def delete_database(db_name: str) -> str:
            \"\"\"Delete an entire database.\"\"\"
            return f"Deleted {db_name}"

        # 2. Register a safety policy hook that explicitly approves execution
        def safety_policy(context: ToolCallHookContext) -> bool | None:
            # Your approval logic here (e.g. prompt user, check allow-list)
            if context.tool_name == "delete_database":
                return True  # Explicitly allow
            return None  # Neutral for other tools

        register_before_tool_call_hook(safety_policy)
    """

    def _make_mock_tools(self, unsafe: bool = False):
        """Helper to create a mock tool setup for execute_tool_and_check_finality.

        Note: AgentAction.tool_input must be a JSON string (not a dict) because
        ToolUsage._validate_tool_input parses it as JSON text.
        """
        from pydantic import BaseModel

        from crewai.agents.parser import AgentAction
        from crewai.utilities.i18n import I18N

        class Schema(BaseModel):
            x: str = "default"

        structured = CrewStructuredTool(
            name="test_tool",
            description="A test tool",
            args_schema=Schema,
            func=lambda x="default": f"executed with {x}",
            unsafe=unsafe,
        )

        agent_action = AgentAction(
            tool="test_tool",
            tool_input='{"x": "value"}',
            text='Action: test_tool\nAction Input: {"x": "value"}',
            thought="Let me use the test tool",
        )

        i18n = I18N()

        return structured, agent_action, i18n

    def test_unsafe_tool_blocked_no_hooks(self):
        """Unsafe tool is denied when no before_tool_call hooks are registered."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=True)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        assert isinstance(result, ToolResult)
        assert "Unsafe tool execution denied (fail-closed)" in result.result
        assert "test_tool" in result.result
        assert result.result_as_answer is False

    def test_unsafe_tool_blocked_hook_returns_none(self):
        """Unsafe tool is denied when hook returns None (neutral, not explicit approval)."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=True)

        def neutral_hook(context: ToolCallHookContext):
            return None  # Neutral - does NOT count as approval

        register_before_tool_call_hook(neutral_hook)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        assert "Unsafe tool execution denied (fail-closed)" in result.result

    def test_unsafe_tool_allowed_hook_returns_true(self):
        """Unsafe tool executes when a hook explicitly returns True."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=True)

        def allow_hook(context: ToolCallHookContext):
            return True  # Explicit approval

        register_before_tool_call_hook(allow_hook)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        # Tool should have executed successfully
        assert "Unsafe tool execution denied" not in result.result
        assert "executed with" in result.result

    def test_safe_tool_unaffected_no_hooks(self):
        """Safe tools (default unsafe=False) execute normally without hooks."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=False)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        assert "executed with" in result.result
        assert "Unsafe tool execution denied" not in result.result

    def test_safe_tool_still_blocked_by_hook_returning_false(self):
        """Safe tools are still blocked if a hook explicitly returns False (existing behavior)."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=False)

        def block_hook(context: ToolCallHookContext):
            return False  # Block execution

        register_before_tool_call_hook(block_hook)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        assert "Tool execution blocked by hook" in result.result

    def test_unsafe_tool_blocked_by_false_even_with_prior_true(self):
        """If a hook returns False, the tool is blocked even if a prior hook returned True."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=True)

        def allow_hook(context: ToolCallHookContext):
            return True

        def block_hook(context: ToolCallHookContext):
            return False

        register_before_tool_call_hook(allow_hook)
        register_before_tool_call_hook(block_hook)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        # False takes priority - but since True was already set, the hook_blocked
        # message applies. The key point is the tool does NOT execute.
        assert "blocked" in result.result.lower() or "denied" in result.result.lower()
        assert "executed with" not in result.result

    def test_error_message_is_deterministic(self):
        """Error message for unsafe tool denial is deterministic and contains actionable info."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=True)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        expected_prefix = "Unsafe tool execution denied (fail-closed):"
        assert result.result.startswith(expected_prefix)
        assert "before_tool_call hook" in result.result
        assert "returning True" in result.result

    def test_multiple_hooks_first_true_allows_unsafe(self):
        """Multiple hooks: if any returns True before a False, tool is allowed."""
        structured, agent_action, i18n = self._make_mock_tools(unsafe=True)

        def allow_hook(context: ToolCallHookContext):
            return True

        def neutral_hook(context: ToolCallHookContext):
            return None

        register_before_tool_call_hook(allow_hook)
        register_before_tool_call_hook(neutral_hook)

        result = execute_tool_and_check_finality(
            agent_action=agent_action,
            tools=[structured],
            i18n=i18n,
        )

        assert "executed with" in result.result
