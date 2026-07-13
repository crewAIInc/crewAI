"""Unit tests for LLM hooks functionality."""

from __future__ import annotations

from unittest.mock import Mock

from crewai.hooks import (
    clear_all_llm_call_hooks,
    unregister_after_llm_call_hook,
    unregister_before_llm_call_hook,
)
from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
    register_after_llm_call_hook,
    register_before_llm_call_hook,
)
import pytest


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
    from crewai.hooks import llm_hooks

    original_before = llm_hooks._before_llm_call_hooks.copy()
    original_after = llm_hooks._after_llm_call_hooks.copy()

    llm_hooks._before_llm_call_hooks.clear()
    llm_hooks._after_llm_call_hooks.clear()

    yield

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

        new_message = {"role": "user", "content": "New message"}
        context.messages.append(new_message)

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

        result = context.response
        for hook in hooks:
            context.response = result
            modified = hook(context)
            if modified is not None:
                result = modified

        assert result == "Original [hook1] [hook2]"

    def test_after_hooks_do_not_clobber_native_tool_call_responses(
        self, mock_executor
    ):
        """A registered after hook must not break native tool execution.

        Regression for crewAIInc/crewAI#6529: `_setup_after_llm_call_hooks`
        stringified structured tool-call payloads, so the executor treated the
        raw tool call as the final answer and never executed the tool. Non-str,
        non-BaseModel responses now pass through untouched; hooks still fire on
        textual responses.
        """
        from crewai.utilities.agent_utils import _setup_after_llm_call_hooks

        observed = []

        def observer(context):
            observed.append(context.response)
            return None

        register_after_llm_call_hook(observer)
        mock_executor.after_llm_call_hooks = get_after_llm_call_hooks()

        tool_calls = [Mock()]  # structured native tool-call payload
        result = _setup_after_llm_call_hooks(
            mock_executor, tool_calls, printer=Mock(), verbose=False
        )
        assert result is tool_calls

        text = _setup_after_llm_call_hooks(
            mock_executor, "final answer", printer=Mock(), verbose=False
        )
        assert text == "final answer"
        assert observed == ["final answer"]

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

    def test_raising_before_hook_does_not_skip_later_hooks(self, mock_executor):
        """Fail-open is per-hook: a crashing hook must not disable its neighbors.

        Regression guard for the dispatcher migration: previously the
        ``except Exception`` wrapped the whole hook loop, so a raising hook
        silently skipped every hook registered after it. Now swallowing is
        per-hook — later hooks still run and the LLM call still proceeds.
        """
        from crewai.utilities.agent_utils import _setup_before_llm_call_hooks

        ran: list[str] = []

        def crashing_hook(context):
            ran.append("crashing")
            raise ValueError("bug in user hook")

        def later_hook(context):
            ran.append("later")

        register_before_llm_call_hook(crashing_hook)
        register_before_llm_call_hook(later_hook)
        mock_executor.before_llm_call_hooks = get_before_llm_call_hooks()

        proceed = _setup_before_llm_call_hooks(
            mock_executor, printer=Mock(), verbose=False
        )

        assert ran == ["crashing", "later"]
        assert proceed is True

    def test_intentional_block_still_short_circuits_later_hooks(self, mock_executor):
        """A hook returning False blocks the call and skips later hooks (unchanged)."""
        from crewai.utilities.agent_utils import _setup_before_llm_call_hooks

        ran: list[str] = []

        def blocking_hook(context):
            ran.append("blocking")
            return False

        def later_hook(context):
            ran.append("later")

        register_before_llm_call_hook(blocking_hook)
        register_before_llm_call_hook(later_hook)
        mock_executor.before_llm_call_hooks = get_before_llm_call_hooks()

        proceed = _setup_before_llm_call_hooks(
            mock_executor, printer=Mock(), verbose=False
        )

        assert ran == ["blocking"]
        assert proceed is False

    @pytest.mark.vcr()
    def test_lite_agent_hooks_integration_with_real_llm(self):
        """Test that LiteAgent executes before/after LLM call hooks and prints messages correctly."""
        import os

        from crewai.lite_agent import LiteAgent

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real LLM test")

        hook_calls = {"before": [], "after": []}

        def before_llm_call_hook(context: LLMCallHookContext) -> bool:
            """Log and verify before hook execution."""
            print(f"\n[BEFORE HOOK] Agent: {context.agent.role if context.agent else 'None'}")
            print(f"[BEFORE HOOK] Iterations: {context.iterations}")
            print(f"[BEFORE HOOK] Message count: {len(context.messages)}")
            print(f"[BEFORE HOOK] Messages: {context.messages}")

            hook_calls["before"].append({
                "iterations": context.iterations,
                "message_count": len(context.messages),
                "has_task": context.task is not None,
                "has_crew": context.crew is not None,
            })

            return True

        def after_llm_call_hook(context: LLMCallHookContext) -> str | None:
            """Log and verify after hook execution."""
            print(f"\n[AFTER HOOK] Agent: {context.agent.role if context.agent else 'None'}")
            print(f"[AFTER HOOK] Iterations: {context.iterations}")
            print(f"[AFTER HOOK] Response: {context.response[:100] if context.response else 'None'}...")
            print(f"[AFTER HOOK] Final message count: {len(context.messages)}")

            hook_calls["after"].append({
                "iterations": context.iterations,
                "has_response": context.response is not None,
                "response_length": len(context.response) if context.response else 0,
            })

            if context.response:
                return f"[HOOKED] {context.response}"
            return None

        register_before_llm_call_hook(before_llm_call_hook)
        register_after_llm_call_hook(after_llm_call_hook)

        try:
            lite_agent = LiteAgent(
                role="Test Assistant",
                goal="Answer questions briefly",
                backstory="You are a helpful test assistant",
                verbose=True,
            )

            assert len(lite_agent.before_llm_call_hooks) > 0, "Before hooks not loaded"
            assert len(lite_agent.after_llm_call_hooks) > 0, "After hooks not loaded"

            result = lite_agent.kickoff("Say 'Hello World' and nothing else")


            assert len(hook_calls["before"]) > 0, "Before hook was never called"
            assert len(hook_calls["after"]) > 0, "After hook was never called"

            # LiteAgent doesn't have task/crew context, unlike agents in CrewBase
            before_call = hook_calls["before"][0]
            assert before_call["has_task"] is False, "Task should be None for LiteAgent in flows"
            assert before_call["has_crew"] is False, "Crew should be None for LiteAgent in flows"
            assert before_call["message_count"] > 0, "Should have messages"

            after_call = hook_calls["after"][0]
            assert after_call["has_response"] is True, "After hook should have response"
            assert after_call["response_length"] > 0, "Response should not be empty"

            # Note: The hook modifies the raw LLM response, but LiteAgent then parses it
            # to extract the "Final Answer" portion. We check the messages to see the modification.
            assert len(result.messages) > 2, "Should have assistant message in messages"
            last_message = result.messages[-1]
            assert last_message["role"] == "assistant", "Last message should be from assistant"
            assert "[HOOKED]" in last_message["content"], "Hook should have modified the assistant message"


        finally:
            unregister_before_llm_call_hook(before_llm_call_hook)
            unregister_after_llm_call_hook(after_llm_call_hook)

    @pytest.mark.vcr()
    def test_direct_llm_call_hooks_integration(self):
        """Test that hooks work for direct llm.call() without agents."""
        import os

        from crewai.llm import LLM

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real LLM test")

        hook_calls = {"before": [], "after": []}

        def before_hook(context: LLMCallHookContext) -> bool:
            """Log and verify before hook execution."""
            print(f"\n[BEFORE HOOK] Agent: {context.agent}")
            print(f"[BEFORE HOOK] Task: {context.task}")
            print(f"[BEFORE HOOK] Crew: {context.crew}")
            print(f"[BEFORE HOOK] LLM: {context.llm}")
            print(f"[BEFORE HOOK] Iterations: {context.iterations}")
            print(f"[BEFORE HOOK] Message count: {len(context.messages)}")

            hook_calls["before"].append({
                "agent": context.agent,
                "task": context.task,
                "crew": context.crew,
                "llm": context.llm is not None,
                "message_count": len(context.messages),
            })

            return True

        def after_hook(context: LLMCallHookContext) -> str | None:
            """Log and verify after hook execution."""
            print(f"\n[AFTER HOOK] Agent: {context.agent}")
            print(f"[AFTER HOOK] Response: {context.response[:100] if context.response else 'None'}...")

            hook_calls["after"].append({
                "has_response": context.response is not None,
                "response_length": len(context.response) if context.response else 0,
            })

            if context.response:
                return f"[HOOKED] {context.response}"
            return None

        register_before_llm_call_hook(before_hook)
        register_after_llm_call_hook(after_hook)

        try:
            llm = LLM(model="gpt-4o-mini")
            result = llm.call([{"role": "user", "content": "Say hello"}])

            print(f"\n[TEST] Final result: {result}")

            assert len(hook_calls["before"]) > 0, "Before hook was never called"
            assert len(hook_calls["after"]) > 0, "After hook was never called"

            before_call = hook_calls["before"][0]
            assert before_call["agent"] is None, "Agent should be None for direct LLM calls"
            assert before_call["task"] is None, "Task should be None for direct LLM calls"
            assert before_call["crew"] is None, "Crew should be None for direct LLM calls"
            assert before_call["llm"] is True, "LLM should be present"
            assert before_call["message_count"] > 0, "Should have messages"

            after_call = hook_calls["after"][0]
            assert after_call["has_response"] is True, "After hook should have response"
            assert after_call["response_length"] > 0, "Response should not be empty"

            assert "[HOOKED]" in result, "Response should be modified by after hook"

        finally:
            unregister_before_llm_call_hook(before_hook)
            unregister_after_llm_call_hook(after_hook)


class TestDirectLLMScopedHooks:
    """Direct (agent-less) LLM calls must honor execution-scoped hooks.

    Regression: the direct-call helpers used to short-circuit when the global
    hook list was empty, so hooks registered only for the current
    ``scoped_hooks()`` context never ran on this path.
    """

    @staticmethod
    def _stub_llm():
        from crewai.llms.base_llm import BaseLLM

        class _StubLLM(BaseLLM):
            def call(self, *args: object, **kwargs: object) -> str:
                return ""

        return _StubLLM(model="stub")

    def test_scoped_before_hook_runs_on_direct_call(self):
        from crewai.hooks import InterceptionPoint
        from crewai.hooks.dispatch import register_scoped, scoped_hooks

        llm = self._stub_llm()
        seen: list[int] = []

        with scoped_hooks():
            register_scoped(
                InterceptionPoint.PRE_MODEL_CALL,
                lambda ctx: seen.append(len(ctx.messages)),
            )
            proceed = llm._invoke_before_llm_call_hooks(
                [{"role": "user", "content": "hi"}], from_agent=None
            )

        assert proceed is True
        assert seen == [1]

    def test_scoped_before_hook_can_block_direct_call(self):
        from crewai.hooks import InterceptionPoint
        from crewai.hooks.dispatch import HookAborted, register_scoped, scoped_hooks

        llm = self._stub_llm()

        def block(ctx: LLMCallHookContext) -> None:
            raise HookAborted(reason="blocked by scoped hook")

        with scoped_hooks():
            register_scoped(InterceptionPoint.PRE_MODEL_CALL, block)
            proceed = llm._invoke_before_llm_call_hooks(
                [{"role": "user", "content": "hi"}], from_agent=None
            )

        assert proceed is False

    def test_scoped_after_hook_modifies_direct_response(self):
        from crewai.hooks import InterceptionPoint
        from crewai.hooks.dispatch import register_scoped, scoped_hooks

        llm = self._stub_llm()

        def redact(ctx: LLMCallHookContext) -> str:
            return ctx.response.replace("SECRET", "[REDACTED]")

        with scoped_hooks():
            register_scoped(InterceptionPoint.POST_MODEL_CALL, redact)
            result = llm._invoke_after_llm_call_hooks(
                [{"role": "user", "content": "hi"}],
                "contains SECRET",
                from_agent=None,
            )

        assert result == "contains [REDACTED]"
