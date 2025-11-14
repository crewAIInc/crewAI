"""Unit tests for LLM hooks functionality."""

from __future__ import annotations

from unittest.mock import Mock

from crewai.hooks import clear_all_llm_call_hooks, unregister_after_llm_call_hook, unregister_before_llm_call_hook
import pytest

from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
    register_after_llm_call_hook,
    register_before_llm_call_hook,
)


@pytest.fixture
def mock_executor():
    """Create a mock executor for testing."""
    executor = Mock()
    executor.messages = [{"role": "system", "content": "Test message"}]
    executor.agent = Mock(role="Test Agent")
    executor.task = Mock(description="Test Task")
    executor.crew = Mock()
    executor.llm = Mock()
    executor.iterations = 0
    return executor


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    # Import the private variables to clear them
    from crewai.hooks import llm_hooks

    # Store original hooks
    original_before = llm_hooks._before_llm_call_hooks.copy()
    original_after = llm_hooks._after_llm_call_hooks.copy()

    # Clear hooks
    llm_hooks._before_llm_call_hooks.clear()
    llm_hooks._after_llm_call_hooks.clear()

    yield

    # Restore original hooks
    llm_hooks._before_llm_call_hooks.clear()
    llm_hooks._after_llm_call_hooks.clear()
    llm_hooks._before_llm_call_hooks.extend(original_before)
    llm_hooks._after_llm_call_hooks.extend(original_after)


class TestLLMCallHookContext:
    """Test LLMCallHookContext initialization and attributes."""

    def test_context_initialization(self, mock_executor):
        """Test that context is initialized correctly with executor."""
        context = LLMCallHookContext(executor=mock_executor)

        assert context.executor == mock_executor
        assert context.messages == mock_executor.messages
        assert context.agent == mock_executor.agent
        assert context.task == mock_executor.task
        assert context.crew == mock_executor.crew
        assert context.llm == mock_executor.llm
        assert context.iterations == mock_executor.iterations
        assert context.response is None

    def test_context_with_response(self, mock_executor):
        """Test that context includes response when provided."""
        test_response = "Test LLM response"
        context = LLMCallHookContext(executor=mock_executor, response=test_response)

        assert context.response == test_response

    def test_messages_are_mutable_reference(self, mock_executor):
        """Test that modifying context.messages modifies executor.messages."""
        context = LLMCallHookContext(executor=mock_executor)

        # Add a message through context
        new_message = {"role": "user", "content": "New message"}
        context.messages.append(new_message)

        # Check that executor.messages is also modified
        assert new_message in mock_executor.messages
        assert len(mock_executor.messages) == 2


class TestBeforeLLMCallHooks:
    """Test before_llm_call hook registration and execution."""

    def test_register_before_hook(self):
        """Test that before hooks are registered correctly."""

        def test_hook(context):
            pass

        register_before_llm_call_hook(test_hook)
        hooks = get_before_llm_call_hooks()

        assert len(hooks) == 1
        assert hooks[0] == test_hook

    def test_multiple_before_hooks(self):
        """Test that multiple before hooks can be registered."""

        def hook1(context):
            pass

        def hook2(context):
            pass

        register_before_llm_call_hook(hook1)
        register_before_llm_call_hook(hook2)
        hooks = get_before_llm_call_hooks()

        assert len(hooks) == 2
        assert hook1 in hooks
        assert hook2 in hooks

    def test_before_hook_can_modify_messages(self, mock_executor):
        """Test that before hooks can modify messages in-place."""

        def add_message_hook(context):
            context.messages.append({"role": "system", "content": "Added by hook"})

        context = LLMCallHookContext(executor=mock_executor)
        add_message_hook(context)

        assert len(context.messages) == 2
        assert context.messages[1]["content"] == "Added by hook"

    def test_get_before_hooks_returns_copy(self):
        """Test that get_before_llm_call_hooks returns a copy."""

        def test_hook(context):
            pass

        register_before_llm_call_hook(test_hook)
        hooks1 = get_before_llm_call_hooks()
        hooks2 = get_before_llm_call_hooks()

        # They should be equal but not the same object
        assert hooks1 == hooks2
        assert hooks1 is not hooks2


class TestAfterLLMCallHooks:
    """Test after_llm_call hook registration and execution."""

    def test_register_after_hook(self):
        """Test that after hooks are registered correctly."""

        def test_hook(context):
            return None

        register_after_llm_call_hook(test_hook)
        hooks = get_after_llm_call_hooks()

        assert len(hooks) == 1
        assert hooks[0] == test_hook

    def test_multiple_after_hooks(self):
        """Test that multiple after hooks can be registered."""

        def hook1(context):
            return None

        def hook2(context):
            return None

        register_after_llm_call_hook(hook1)
        register_after_llm_call_hook(hook2)
        hooks = get_after_llm_call_hooks()

        assert len(hooks) == 2
        assert hook1 in hooks
        assert hook2 in hooks

    def test_after_hook_can_modify_response(self, mock_executor):
        """Test that after hooks can modify the response."""
        original_response = "Original response"

        def modify_response_hook(context):
            if context.response:
                return context.response.replace("Original", "Modified")
            return None

        context = LLMCallHookContext(executor=mock_executor, response=original_response)
        modified = modify_response_hook(context)

        assert modified == "Modified response"

    def test_after_hook_returns_none_keeps_original(self, mock_executor):
        """Test that returning None keeps the original response."""
        original_response = "Original response"

        def no_change_hook(context):
            return None

        context = LLMCallHookContext(executor=mock_executor, response=original_response)
        result = no_change_hook(context)

        assert result is None
        assert context.response == original_response

    def test_get_after_hooks_returns_copy(self):
        """Test that get_after_llm_call_hooks returns a copy."""

        def test_hook(context):
            return None

        register_after_llm_call_hook(test_hook)
        hooks1 = get_after_llm_call_hooks()
        hooks2 = get_after_llm_call_hooks()

        # They should be equal but not the same object
        assert hooks1 == hooks2
        assert hooks1 is not hooks2


class TestLLMHooksIntegration:
    """Test integration scenarios with multiple hooks."""

    def test_multiple_before_hooks_execute_in_order(self, mock_executor):
        """Test that multiple before hooks execute in registration order."""
        execution_order = []

        def hook1(context):
            execution_order.append(1)

        def hook2(context):
            execution_order.append(2)

        def hook3(context):
            execution_order.append(3)

        register_before_llm_call_hook(hook1)
        register_before_llm_call_hook(hook2)
        register_before_llm_call_hook(hook3)

        context = LLMCallHookContext(executor=mock_executor)
        hooks = get_before_llm_call_hooks()

        for hook in hooks:
            hook(context)

        assert execution_order == [1, 2, 3]

    def test_multiple_after_hooks_chain_modifications(self, mock_executor):
        """Test that multiple after hooks can chain modifications."""

        def hook1(context):
            if context.response:
                return context.response + " [hook1]"
            return None

        def hook2(context):
            if context.response:
                return context.response + " [hook2]"
            return None

        register_after_llm_call_hook(hook1)
        register_after_llm_call_hook(hook2)

        context = LLMCallHookContext(executor=mock_executor, response="Original")
        hooks = get_after_llm_call_hooks()

        # Simulate chaining (how it would be used in practice)
        result = context.response
        for hook in hooks:
            # Update context for next hook
            context.response = result
            modified = hook(context)
            if modified is not None:
                result = modified

        assert result == "Original [hook1] [hook2]"

    def test_unregister_before_hook(self):
        """Test that before hooks can be unregistered."""
        def test_hook(context):
            pass

        register_before_llm_call_hook(test_hook)
        unregister_before_llm_call_hook(test_hook)
        hooks = get_before_llm_call_hooks()
        assert len(hooks) == 0

    def test_unregister_after_hook(self):
        """Test that after hooks can be unregistered."""
        def test_hook(context):
            return None

        register_after_llm_call_hook(test_hook)
        unregister_after_llm_call_hook(test_hook)
        hooks = get_after_llm_call_hooks()
        assert len(hooks) == 0

    def test_clear_all_llm_call_hooks(self):
        """Test that all llm call hooks can be cleared."""
        def test_hook(context):
            pass

        register_before_llm_call_hook(test_hook)
        register_after_llm_call_hook(test_hook)
        clear_all_llm_call_hooks()
        hooks = get_before_llm_call_hooks()
        assert len(hooks) == 0
