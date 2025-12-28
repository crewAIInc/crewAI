from __future__ import annotations

from unittest.mock import Mock

from crewai.hooks import clear_all_tool_call_hooks, unregister_after_tool_call_hook, unregister_before_tool_call_hook
import pytest

from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
)


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = Mock()
    tool.name = "test_tool"
    tool.description = "Test tool description"
    return tool


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.role = "Test Agent"
    return agent


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    task = Mock()
    task.description = "Test task"
    return task


@pytest.fixture
def mock_crew():
    """Create a mock crew for testing."""
    crew = Mock()
    return crew


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    from crewai.hooks import tool_hooks

    # Store original hooks
    original_before = tool_hooks._before_tool_call_hooks.copy()
    original_after = tool_hooks._after_tool_call_hooks.copy()

    # Clear hooks
    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()

    yield

    # Restore original hooks
    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.extend(original_before)
    tool_hooks._after_tool_call_hooks.extend(original_after)


class TestToolCallHookContext:
    """Test ToolCallHookContext initialization and attributes."""

    def test_context_initialization(self, mock_tool, mock_agent, mock_task, mock_crew):
        """Test that context is initialized correctly."""
        tool_input = {"arg1": "value1", "arg2": "value2"}

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            agent=mock_agent,
            task=mock_task,
            crew=mock_crew,
        )

        assert context.tool_name == "test_tool"
        assert context.tool_input == tool_input
        assert context.tool == mock_tool
        assert context.agent == mock_agent
        assert context.task == mock_task
        assert context.crew == mock_crew
        assert context.tool_result is None

    def test_context_with_result(self, mock_tool):
        """Test that context includes result when provided."""
        tool_input = {"arg1": "value1"}
        tool_result = "Test tool result"

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result=tool_result,
        )

        assert context.tool_result == tool_result

    def test_tool_input_is_mutable_reference(self, mock_tool):
        """Test that modifying context.tool_input modifies the original dict."""
        tool_input = {"arg1": "value1"}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        # Modify through context
        context.tool_input["arg2"] = "value2"

        # Check that original dict is also modified
        assert "arg2" in tool_input
        assert tool_input["arg2"] == "value2"


class TestBeforeToolCallHooks:
    """Test before_tool_call hook registration and execution."""

    def test_register_before_hook(self):
        """Test that before hooks are registered correctly."""
        def test_hook(context):
            return None

        register_before_tool_call_hook(test_hook)
        hooks = get_before_tool_call_hooks()

        assert len(hooks) == 1
        assert hooks[0] == test_hook

    def test_multiple_before_hooks(self):
        """Test that multiple before hooks can be registered."""
        def hook1(context):
            return None

        def hook2(context):
            return None

        register_before_tool_call_hook(hook1)
        register_before_tool_call_hook(hook2)
        hooks = get_before_tool_call_hooks()

        assert len(hooks) == 2
        assert hook1 in hooks
        assert hook2 in hooks

    def test_before_hook_can_block_execution(self, mock_tool):
        """Test that before hooks can block tool execution."""
        def block_hook(context):
            if context.tool_name == "dangerous_tool":
                return False  # Block execution
            return None  # Allow execution

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="dangerous_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        result = block_hook(context)
        assert result is False

    def test_before_hook_can_allow_execution(self, mock_tool):
        """Test that before hooks can explicitly allow execution."""
        def allow_hook(context):
            return None  # Allow execution

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="safe_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        result = allow_hook(context)
        assert result is None

    def test_before_hook_can_modify_input(self, mock_tool):
        """Test that before hooks can modify tool input in-place."""
        def modify_input_hook(context):
            context.tool_input["modified_by_hook"] = True
            return None

        tool_input = {"arg1": "value1"}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        modify_input_hook(context)

        assert "modified_by_hook" in context.tool_input
        assert context.tool_input["modified_by_hook"] is True

    def test_get_before_hooks_returns_copy(self):
        """Test that get_before_tool_call_hooks returns a copy."""
        def test_hook(context):
            return None

        register_before_tool_call_hook(test_hook)
        hooks1 = get_before_tool_call_hooks()
        hooks2 = get_before_tool_call_hooks()

        # They should be equal but not the same object
        assert hooks1 == hooks2
        assert hooks1 is not hooks2


class TestAfterToolCallHooks:
    """Test after_tool_call hook registration and execution."""

    def test_register_after_hook(self):
        """Test that after hooks are registered correctly."""
        def test_hook(context):
            return None

        register_after_tool_call_hook(test_hook)
        hooks = get_after_tool_call_hooks()

        assert len(hooks) == 1
        assert hooks[0] == test_hook

    def test_multiple_after_hooks(self):
        """Test that multiple after hooks can be registered."""
        def hook1(context):
            return None

        def hook2(context):
            return None

        register_after_tool_call_hook(hook1)
        register_after_tool_call_hook(hook2)
        hooks = get_after_tool_call_hooks()

        assert len(hooks) == 2
        assert hook1 in hooks
        assert hook2 in hooks

    def test_after_hook_can_modify_result(self, mock_tool):
        """Test that after hooks can modify the tool result."""
        original_result = "Original result"

        def modify_result_hook(context):
            if context.tool_result:
                return context.tool_result.replace("Original", "Modified")
            return None

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result=original_result,
        )

        modified = modify_result_hook(context)
        assert modified == "Modified result"

    def test_after_hook_returns_none_keeps_original(self, mock_tool):
        """Test that returning None keeps the original result."""
        original_result = "Original result"

        def no_change_hook(context):
            return None

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result=original_result,
        )

        result = no_change_hook(context)

        assert result is None
        assert context.tool_result == original_result

    def test_get_after_hooks_returns_copy(self):
        """Test that get_after_tool_call_hooks returns a copy."""
        def test_hook(context):
            return None

        register_after_tool_call_hook(test_hook)
        hooks1 = get_after_tool_call_hooks()
        hooks2 = get_after_tool_call_hooks()

        # They should be equal but not the same object
        assert hooks1 == hooks2
        assert hooks1 is not hooks2


class TestToolHooksIntegration:
    """Test integration scenarios with multiple hooks."""

    def test_multiple_before_hooks_execute_in_order(self, mock_tool):
        """Test that multiple before hooks execute in registration order."""
        execution_order = []

        def hook1(context):
            execution_order.append(1)
            return None

        def hook2(context):
            execution_order.append(2)
            return None

        def hook3(context):
            execution_order.append(3)
            return None

        register_before_tool_call_hook(hook1)
        register_before_tool_call_hook(hook2)
        register_before_tool_call_hook(hook3)

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        for hook in hooks:
            hook(context)

        assert execution_order == [1, 2, 3]

    def test_first_blocking_hook_stops_execution(self, mock_tool):
        """Test that first hook returning False blocks execution."""
        execution_order = []

        def hook1(context):
            execution_order.append(1)
            return None  # Allow

        def hook2(context):
            execution_order.append(2)
            return False  # Block

        def hook3(context):
            execution_order.append(3)
            return None  # This shouldn't run

        register_before_tool_call_hook(hook1)
        register_before_tool_call_hook(hook2)
        register_before_tool_call_hook(hook3)

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        blocked = False
        for hook in hooks:
            result = hook(context)
            if result is False:
                blocked = True
                break

        assert blocked is True
        assert execution_order == [1, 2]  # hook3 didn't run

    def test_multiple_after_hooks_chain_modifications(self, mock_tool):
        """Test that multiple after hooks can chain modifications."""
        def hook1(context):
            if context.tool_result:
                return context.tool_result + " [hook1]"
            return None

        def hook2(context):
            if context.tool_result:
                return context.tool_result + " [hook2]"
            return None

        register_after_tool_call_hook(hook1)
        register_after_tool_call_hook(hook2)

        tool_input = {}
        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input=tool_input,
            tool=mock_tool,
            tool_result="Original",
        )

        hooks = get_after_tool_call_hooks()

        # Simulate chaining (how it would be used in practice)
        result = context.tool_result
        for hook in hooks:
            # Update context for next hook
            context.tool_result = result
            modified = hook(context)
            if modified is not None:
                result = modified

        assert result == "Original [hook1] [hook2]"

    def test_hooks_with_validation_and_sanitization(self, mock_tool):
        """Test a realistic scenario with validation and sanitization hooks."""
        # Validation hook (before)
        def validate_file_path(context):
            if context.tool_name == "write_file":
                file_path = context.tool_input.get("file_path", "")
                if ".env" in file_path:
                    return False  # Block sensitive files
            return None

        # Sanitization hook (after)
        def sanitize_secrets(context):
            if context.tool_result and "SECRET_KEY" in context.tool_result:
                return context.tool_result.replace("SECRET_KEY=abc123", "SECRET_KEY=[REDACTED]")
            return None

        register_before_tool_call_hook(validate_file_path)
        register_after_tool_call_hook(sanitize_secrets)

        # Test blocking
        blocked_context = ToolCallHookContext(
            tool_name="write_file",
            tool_input={"file_path": ".env"},
            tool=mock_tool,
        )

        before_hooks = get_before_tool_call_hooks()
        blocked = False
        for hook in before_hooks:
            if hook(blocked_context) is False:
                blocked = True
                break

        assert blocked is True

        # Test sanitization
        sanitize_context = ToolCallHookContext(
            tool_name="read_file",
            tool_input={"file_path": "config.txt"},
            tool=mock_tool,
            tool_result="Content: SECRET_KEY=abc123",
        )

        after_hooks = get_after_tool_call_hooks()
        result = sanitize_context.tool_result
        for hook in after_hooks:
            sanitize_context.tool_result = result
            modified = hook(sanitize_context)
            if modified is not None:
                result = modified

        assert "SECRET_KEY=[REDACTED]" in result
        assert "abc123" not in result


    def test_unregister_before_hook(self):
        """Test that before hooks can be unregistered."""
        def test_hook(context):
            pass

        register_before_tool_call_hook(test_hook)
        unregister_before_tool_call_hook(test_hook)
        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 0

    def test_unregister_after_hook(self):
        """Test that after hooks can be unregistered."""
        def test_hook(context):
            return None

        register_after_tool_call_hook(test_hook)
        unregister_after_tool_call_hook(test_hook)
        hooks = get_after_tool_call_hooks()
        assert len(hooks) == 0

    def test_clear_all_tool_call_hooks(self):
        """Test that all tool call hooks can be cleared."""
        def test_hook(context):
            pass

        register_before_tool_call_hook(test_hook)
        register_after_tool_call_hook(test_hook)
        clear_all_tool_call_hooks()
        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 0

    @pytest.mark.vcr()
    def test_lite_agent_hooks_integration_with_real_tool(self):
        """Test that LiteAgent executes before/after tool call hooks with real tool calls."""
        import os
        from crewai.lite_agent import LiteAgent
        from crewai.tools import tool

        # Skip if no API key available
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real tool test")

        # Track hook invocations
        hook_calls = {"before": [], "after": []}

        # Create a simple test tool
        @tool("calculate_sum")
        def calculate_sum(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        def before_tool_call_hook(context: ToolCallHookContext) -> bool:
            """Log and verify before hook execution."""
            print(f"\n[BEFORE HOOK] Tool: {context.tool_name}")
            print(f"[BEFORE HOOK] Tool input: {context.tool_input}")
            print(f"[BEFORE HOOK] Agent: {context.agent.role if context.agent else 'None'}")
            print(f"[BEFORE HOOK] Task: {context.task}")
            print(f"[BEFORE HOOK] Crew: {context.crew}")

            # Track the call
            hook_calls["before"].append({
                "tool_name": context.tool_name,
                "tool_input": context.tool_input,
                "has_agent": context.agent is not None,
                "has_task": context.task is not None,
                "has_crew": context.crew is not None,
            })

            return True  # Allow execution

        def after_tool_call_hook(context: ToolCallHookContext) -> str | None:
            """Log and verify after hook execution."""
            print(f"\n[AFTER HOOK] Tool: {context.tool_name}")
            print(f"[AFTER HOOK] Tool result: {context.tool_result}")
            print(f"[AFTER HOOK] Agent: {context.agent.role if context.agent else 'None'}")

            # Track the call
            hook_calls["after"].append({
                "tool_name": context.tool_name,
                "tool_result": context.tool_result,
                "has_result": context.tool_result is not None,
            })

            return None  # Don't modify result

        # Register hooks
        register_before_tool_call_hook(before_tool_call_hook)
        register_after_tool_call_hook(after_tool_call_hook)

        try:
            # Create LiteAgent with the tool
            lite_agent = LiteAgent(
                role="Calculator Assistant",
                goal="Help with math calculations",
                backstory="You are a helpful calculator assistant",
                tools=[calculate_sum],
                verbose=True,
            )

            # Execute with a prompt that should trigger tool usage
            result = lite_agent.kickoff("What is 5 + 3? Use the calculate_sum tool.")

            # Verify hooks were called
            assert len(hook_calls["before"]) > 0, "Before hook was never called"
            assert len(hook_calls["after"]) > 0, "After hook was never called"

            # Verify context had correct attributes for LiteAgent (used in flows)
            # LiteAgent doesn't have task/crew context, unlike agents in CrewBase
            before_call = hook_calls["before"][0]
            assert before_call["tool_name"] == "calculate_sum", "Tool name should be 'calculate_sum'"
            assert "a" in before_call["tool_input"], "Tool input should have 'a' parameter"
            assert "b" in before_call["tool_input"], "Tool input should have 'b' parameter"

            # Verify after hook received result
            after_call = hook_calls["after"][0]
            assert after_call["has_result"] is True, "After hook should have tool result"
            assert after_call["tool_name"] == "calculate_sum", "Tool name should match"
            # The result should contain the sum (8)
            assert "8" in str(after_call["tool_result"]), "Tool result should contain the sum"

        finally:
            # Clean up hooks
            unregister_before_tool_call_hook(before_tool_call_hook)
            unregister_after_tool_call_hook(after_tool_call_hook)
