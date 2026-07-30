"""Unit tests for LLM hooks functionality."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
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

    def test_scoped_hooks_fire_on_agent_executor_llm_seams(self, mock_executor):
        """register_scoped hooks must run on the executor model seams.

        Regression: `_setup_before/after_llm_call_hooks` only ran the
        executor's snapshot lists, so execution-scoped hooks never fired on
        PRE/POST_MODEL_CALL during normal agent execution (while tool seams,
        which go through `dispatch`, merged them). Scoped hooks run after the
        snapshot, matching dispatch's global-then-scoped ordering.
        """
        from crewai.hooks import InterceptionPoint
        from crewai.hooks.dispatch import register_scoped, scoped_hooks
        from crewai.utilities.agent_utils import (
            _setup_after_llm_call_hooks,
            _setup_before_llm_call_hooks,
        )

        order: list[str] = []

        def snapshot_hook(context):
            order.append("snapshot")

        mock_executor.before_llm_call_hooks = [snapshot_hook]
        mock_executor.after_llm_call_hooks = []

        with scoped_hooks():
            register_scoped(
                InterceptionPoint.PRE_MODEL_CALL,
                lambda ctx: order.append("scoped_pre"),
            )
            register_scoped(
                InterceptionPoint.POST_MODEL_CALL,
                lambda ctx: order.append("scoped_post"),
            )

            proceed = _setup_before_llm_call_hooks(
                mock_executor, printer=Mock(), verbose=False
            )
            answer = _setup_after_llm_call_hooks(
                mock_executor, "answer", printer=Mock(), verbose=False
            )

        assert order == ["snapshot", "scoped_pre", "scoped_post"]
        assert proceed is True
        assert answer == "answer"

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


class _Recorder:
    """Records what each provider client was actually asked to send.

    ``issued`` answers whether a request went out at all (and on which path),
    which is what the blocking tests need. ``payloads`` keeps the request kwargs
    so a test can assert on the data the provider received rather than on the
    in-memory list the hook mutated -- the two are only the same object if the
    guard passes the real payload through, which is the property under test.
    """

    def __init__(self) -> None:
        self.issued: list[str] = []
        self.payloads: list[dict[str, Any]] = []

    def record(self, path: str, kwargs: dict[str, Any]) -> None:
        self.issued.append(path)
        self.payloads.append(kwargs)

    @property
    def payload_text(self) -> str:
        """Every recorded payload flattened to text.

        Providers disagree on payload shape (plain dicts, Anthropic content
        blocks, Gemini ``Content`` objects), and these tests only ask whether a
        hook's addition is in there, so the repr is searched instead of a
        per-provider structure being walked.
        """
        return repr(self.payloads)


def _usage_stub() -> Any:
    """Token-usage payload accepted by every provider's usage extractor."""
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )


def _anthropic_response() -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="the model answered")],
        usage=_usage_stub(),
        stop_reason="end_turn",
        id="msg_stub",
    )


def _openai_response() -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="the model answered", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=_usage_stub(),
        id="cmpl_stub",
    )


def _stub_anthropic_llm(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real AnthropicCompletion whose SDK client records every request."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-sonnet-4-5", api_key="stub", stream=False)

    def create(**kwargs: Any) -> Any:
        rec.record("sync", kwargs)
        return _anthropic_response()

    async def acreate(**kwargs: Any) -> Any:
        rec.record("async", kwargs)
        return _anthropic_response()

    llm._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    llm._async_client = SimpleNamespace(messages=SimpleNamespace(create=acreate))
    return llm


def _stub_openai_llm(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real OpenAICompletion whose SDK client records every request."""
    from crewai.llms.providers.openai.completion import OpenAICompletion

    llm = OpenAICompletion(model="gpt-4o", api_key="stub", stream=False)

    def create(**kwargs: Any) -> Any:
        rec.record("sync", kwargs)
        return _openai_response()

    async def acreate(**kwargs: Any) -> Any:
        rec.record("async", kwargs)
        return _openai_response()

    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    llm._async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=acreate))
    )
    return llm


def _bedrock_response() -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "the model answered"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


def _stub_bedrock_llm(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real BedrockCompletion whose Converse client records every request."""
    from crewai.llms.providers.bedrock import completion as bedrock_completion
    from crewai.llms.providers.bedrock.completion import BedrockCompletion

    # ``acall`` refuses to run at all without aiobotocore, which is an optional
    # extra. The hook gating under test is upstream of any AWS I/O and the client
    # below is a stub, so the dependency check is patched rather than installed.
    monkeypatch.setattr(bedrock_completion, "AIOBOTOCORE_AVAILABLE", True)

    llm = BedrockCompletion(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_access_key_id="stub",
        aws_secret_access_key="stub",
        region_name="us-east-1",
        stream=False,
    )

    def converse(**kwargs: Any) -> dict[str, Any]:
        rec.record("sync", kwargs)
        return _bedrock_response()

    async def aconverse(**kwargs: Any) -> dict[str, Any]:
        rec.record("async", kwargs)
        return _bedrock_response()

    llm._client = SimpleNamespace(converse=converse)
    # ``_ensure_async_client`` builds the aiobotocore client inside an exit stack;
    # marking it initialized makes it return this stub instead.
    llm._async_client = SimpleNamespace(converse=aconverse)
    llm._async_client_initialized = True
    return llm


def _gemini_response() -> Any:
    part = SimpleNamespace(text="the model answered", function_call=None, thought=None)
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[part], role="model"),
                finish_reason=None,
            )
        ],
        text="the model answered",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=1,
            total_token_count=2,
            cached_content_token_count=0,
        ),
        function_calls=None,
    )


def _stub_gemini_llm(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real GeminiCompletion whose genai client records every request.

    Gemini exposes one client for both paths (``_get_async_client`` returns
    ``_get_sync_client``), with the async surface under ``.aio``.
    """
    from crewai.llms.providers.gemini.completion import GeminiCompletion

    llm = GeminiCompletion(model="gemini-2.0-flash", api_key="stub", stream=False)

    def generate_content(**kwargs: Any) -> Any:
        rec.record("sync", kwargs)
        return _gemini_response()

    async def agenerate_content(**kwargs: Any) -> Any:
        rec.record("async", kwargs)
        return _gemini_response()

    llm._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content),
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=agenerate_content)
        ),
        vertexai=False,
    )
    return llm


def _azure_response() -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="the model answered", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=_usage_stub(),
        id="azure_stub",
    )


def _stub_azure_llm(rec: _Recorder, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real AzureCompletion whose inference client records every request."""
    from crewai.llms.providers.azure.completion import AzureCompletion

    llm = AzureCompletion(
        model="gpt-4o",
        api_key="stub",
        endpoint="https://stub.services.ai.azure.com/models",
        stream=False,
    )

    def complete(**kwargs: Any) -> Any:
        rec.record("sync", kwargs)
        return _azure_response()

    async def acomplete(**kwargs: Any) -> Any:
        rec.record("async", kwargs)
        return _azure_response()

    llm._client = SimpleNamespace(complete=complete)
    llm._async_client = SimpleNamespace(complete=acomplete)
    return llm


_PROVIDER_STUBS = {
    "openai": _stub_openai_llm,
    "anthropic": _stub_anthropic_llm,
    "bedrock": _stub_bedrock_llm,
    "gemini": _stub_gemini_llm,
    "azure": _stub_azure_llm,
}


class TestBeforeLLMCallHooksOnAsyncPaths:
    """before_llm_call must gate acall() exactly as it gates call().

    A hook returning False aborts the call. When acall() skipped the hook the
    request still went out, so a policy check that worked synchronously silently
    stopped blocking anything once the caller switched to async.
    """

    @pytest.fixture(params=sorted(_PROVIDER_STUBS))
    def provider(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Any, _Recorder]:
        rec = _Recorder()
        return _PROVIDER_STUBS[request.param](rec, monkeypatch), rec

    def test_sync_call_is_blocked_by_before_hook(
        self, provider: tuple[Any, _Recorder]
    ) -> None:
        llm, rec = provider
        seen: list[int] = []
        register_before_llm_call_hook(lambda ctx: seen.append(len(ctx.messages)) or False)

        with pytest.raises(ValueError, match="blocked by before_llm_call hook"):
            llm.call("hi")

        assert len(seen) == 1
        assert rec.issued == []

    @pytest.mark.asyncio
    async def test_async_call_is_blocked_by_before_hook(
        self, provider: tuple[Any, _Recorder]
    ) -> None:
        """The regression: acall() used to issue the request and return the response."""
        llm, rec = provider
        seen: list[int] = []
        register_before_llm_call_hook(lambda ctx: seen.append(len(ctx.messages)) or False)

        with pytest.raises(ValueError, match="blocked by before_llm_call hook"):
            await llm.acall("hi")

        assert len(seen) == 1
        assert rec.issued == []

    @pytest.mark.asyncio
    async def test_async_call_runs_before_hook_that_allows(
        self, provider: tuple[Any, _Recorder]
    ) -> None:
        """A hook that does not abort observes the messages and the call proceeds."""
        llm, rec = provider
        seen: list[list[dict[str, Any]]] = []

        def observe(ctx: LLMCallHookContext) -> None:
            seen.append(list(ctx.messages))

        register_before_llm_call_hook(observe)

        result = await llm.acall("hi")

        assert result == "the model answered"
        assert rec.issued == ["async"]
        # The content shape differs per provider (a plain string, a list of
        # content blocks, a Gemini part dict), so the prompt is located inside
        # the last message rather than compared to one provider's layout.
        assert seen and "hi" in str(seen[0][-1])
        # The hook must see the prompt the provider is actually about to be sent,
        # not a placeholder assembled for the hook's benefit.
        assert "hi" in rec.payload_text

    @pytest.mark.asyncio
    async def test_async_call_without_hooks_is_unchanged(
        self, provider: tuple[Any, _Recorder]
    ) -> None:
        """Negative control: no hook registered, nothing blocked or altered."""
        llm, rec = provider

        result = await llm.acall("hi")

        assert result == "the model answered"
        assert rec.issued == ["async"]

    @pytest.mark.asyncio
    async def test_async_before_hook_mutation_reaches_the_provider_as_on_sync(
        self, provider: tuple[Any, _Recorder]
    ) -> None:
        """A mutation must land in the request, and to the same extent as on sync.

        Asserting only that ``ctx.messages`` gained an entry would pass even if
        the guard were handed a copy and ``acall()`` dropped the mutation on the
        floor, so the check is against the kwargs the stubbed client received.

        The comparison is against the *sync* path rather than a hard-coded
        expectation because whether a mutation propagates at all is a
        pre-existing per-provider property, not something this change decides:
        four providers pass the payload list straight to the hook, while Gemini
        converts its ``Content`` objects to dicts first and so hands over a copy
        on both paths. Pinning ``async == sync`` states the invariant this PR is
        responsible for -- acall() gates and forwards exactly as call() does --
        and would fail if the async guard were ever wired to a different list
        than its sync twin, without asserting that Gemini's copy is correct.
        """
        llm, rec = provider
        guard_rail = {"role": "user", "content": "and be brief"}

        register_before_llm_call_hook(
            lambda ctx: ctx.messages.append(guard_rail),  # type: ignore[func-returns-value]
        )

        llm.call("hi")
        sync_propagated = "and be brief" in rec.payload_text
        assert rec.issued == ["sync"]

        rec.issued.clear()
        rec.payloads.clear()

        await llm.acall("hi")
        async_propagated = "and be brief" in rec.payload_text

        assert rec.issued == ["async"]
        assert async_propagated == sync_propagated

    @pytest.mark.asyncio
    async def test_async_before_hook_mutation_reaches_the_provider(
        self, provider: tuple[Any, _Recorder]
    ) -> None:
        """The four providers that forward the payload list really do forward it.

        The symmetry test above cannot tell "both paths propagate" from "neither
        does", so it would still pass if a future refactor made every provider
        hand the guard a copy. This pins the propagating providers by name.
        Gemini is excluded because its own sync path does not propagate either;
        that asymmetry is upstream of this change and is left alone.
        """
        llm, rec = provider
        if llm.__class__.__name__ == "GeminiCompletion":
            pytest.skip("Gemini hands the guard a converted copy on both paths")

        register_before_llm_call_hook(
            lambda ctx: ctx.messages.append(  # type: ignore[func-returns-value]
                {"role": "user", "content": "and be brief"}
            ),
        )

        await llm.acall("hi")

        assert rec.issued == ["async"]
        assert "and be brief" in rec.payload_text
