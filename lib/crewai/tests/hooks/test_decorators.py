"""Tests for decorator-based hook registration."""

from __future__ import annotations

from unittest.mock import Mock

from crewai.hooks import (
    GuardrailDecision,
    GuardrailRequest,
    after_llm_call,
    after_tool_call,
    before_llm_call,
    before_tool_call,
    enable_guardrail,
    get_after_llm_call_hooks,
    get_after_tool_call_hooks,
    get_before_llm_call_hooks,
    get_before_tool_call_hooks,
)
from crewai.hooks.llm_hooks import LLMCallHookContext
from crewai.hooks.tool_hooks import ToolCallHookContext
import pytest


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    from crewai.hooks import llm_hooks, tool_hooks

    original_before_llm = llm_hooks._before_llm_call_hooks.copy()
    original_after_llm = llm_hooks._after_llm_call_hooks.copy()
    original_before_tool = tool_hooks._before_tool_call_hooks.copy()
    original_after_tool = tool_hooks._after_tool_call_hooks.copy()

    llm_hooks._before_llm_call_hooks.clear()
    llm_hooks._after_llm_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()

    yield

    llm_hooks._before_llm_call_hooks.clear()
    llm_hooks._after_llm_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()
    llm_hooks._before_llm_call_hooks.extend(original_before_llm)
    llm_hooks._after_llm_call_hooks.extend(original_after_llm)
    tool_hooks._before_tool_call_hooks.extend(original_before_tool)
    tool_hooks._after_tool_call_hooks.extend(original_after_tool)


class TestLLMHookDecorators:
    """Test LLM hook decorators."""

    def test_before_llm_call_decorator_registers_hook(self):
        """Test that @before_llm_call decorator registers the hook."""

        @before_llm_call
        def test_hook(context):
            pass

        hooks = get_before_llm_call_hooks()
        assert len(hooks) == 1

    def test_after_llm_call_decorator_registers_hook(self):
        """Test that @after_llm_call decorator registers the hook."""

        @after_llm_call
        def test_hook(context):
            return None

        hooks = get_after_llm_call_hooks()
        assert len(hooks) == 1

    def test_decorated_hook_executes_correctly(self):
        """Test that decorated hook executes and modifies behavior."""
        execution_log = []

        @before_llm_call
        def test_hook(context):
            execution_log.append("executed")

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Test")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        context = LLMCallHookContext(executor=mock_executor)

        hooks = get_before_llm_call_hooks()
        hooks[0](context)

        assert len(execution_log) == 1
        assert execution_log[0] == "executed"

    def test_before_llm_call_with_agent_filter(self):
        """Test that agent filter works correctly."""
        execution_log = []

        @before_llm_call(agents=["Researcher"])
        def filtered_hook(context):
            execution_log.append(context.agent.role)

        hooks = get_before_llm_call_hooks()
        assert len(hooks) == 1

        mock_executor = Mock()
        mock_executor.messages = []
        mock_executor.agent = Mock(role="Researcher")
        mock_executor.task = Mock()
        mock_executor.crew = Mock()
        mock_executor.llm = Mock()
        mock_executor.iterations = 0

        context = LLMCallHookContext(executor=mock_executor)
        hooks[0](context)

        assert len(execution_log) == 1
        assert execution_log[0] == "Researcher"

        mock_executor.agent.role = "Analyst"
        context2 = LLMCallHookContext(executor=mock_executor)
        hooks[0](context2)

        assert len(execution_log) == 1


class TestToolHookDecorators:
    """Test tool hook decorators."""

    def test_before_tool_call_decorator_registers_hook(self):
        """Test that @before_tool_call decorator registers the hook."""

        @before_tool_call
        def test_hook(context):
            return None

        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 1

    def test_after_tool_call_decorator_registers_hook(self):
        """Test that @after_tool_call decorator registers the hook."""

        @after_tool_call
        def test_hook(context):
            return None

        hooks = get_after_tool_call_hooks()
        assert len(hooks) == 1

    def test_before_tool_call_with_tool_filter(self):
        """Test that tool filter works correctly."""
        execution_log = []

        @before_tool_call(tools=["delete_file", "execute_code"])
        def filtered_hook(context):
            execution_log.append(context.tool_name)
            return

        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 1

        mock_tool = Mock()
        context = ToolCallHookContext(
            tool_name="delete_file",
            tool_input={},
            tool=mock_tool,
        )
        hooks[0](context)

        assert len(execution_log) == 1
        assert execution_log[0] == "delete_file"

        context2 = ToolCallHookContext(
            tool_name="read_file",
            tool_input={},
            tool=mock_tool,
        )
        hooks[0](context2)

        assert len(execution_log) == 1

    def test_before_tool_call_tool_filter_sanitizes_names(self):
        """Tool filter should auto-sanitize names so users can pass BaseTool.name directly."""
        execution_log = []

        @before_tool_call(tools=["Delete File", "Execute Code"])
        def filtered_hook(context):
            execution_log.append(context.tool_name)
            return

        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 1

        mock_tool = Mock()
        context = ToolCallHookContext(
            tool_name="delete_file",
            tool_input={},
            tool=mock_tool,
        )
        hooks[0](context)
        assert execution_log == ["delete_file"]

        context2 = ToolCallHookContext(
            tool_name="read_file",
            tool_input={},
            tool=mock_tool,
        )
        hooks[0](context2)
        assert execution_log == ["delete_file"]

    def test_before_tool_call_with_combined_filters(self):
        """Test that combined tool and agent filters work."""
        execution_log = []

        @before_tool_call(tools=["write_file"], agents=["Developer"])
        def filtered_hook(context):
            execution_log.append(f"{context.tool_name}-{context.agent.role}")
            return

        hooks = get_before_tool_call_hooks()
        mock_tool = Mock()
        mock_agent = Mock(role="Developer")

        context = ToolCallHookContext(
            tool_name="write_file",
            tool_input={},
            tool=mock_tool,
            agent=mock_agent,
        )
        hooks[0](context)

        assert len(execution_log) == 1
        assert execution_log[0] == "write_file-Developer"

        mock_agent.role = "Researcher"
        context2 = ToolCallHookContext(
            tool_name="write_file",
            tool_input={},
            tool=mock_tool,
            agent=mock_agent,
        )
        hooks[0](context2)

        assert len(execution_log) == 1

    def test_after_tool_call_with_filter(self):
        """Test that after_tool_call decorator with filter works."""

        @after_tool_call(tools=["web_search"])
        def filtered_hook(context):
            if context.tool_result:
                return context.tool_result.upper()
            return None

        hooks = get_after_tool_call_hooks()
        mock_tool = Mock()

        context = ToolCallHookContext(
            tool_name="web_search",
            tool_input={},
            tool=mock_tool,
            tool_result="result",
        )
        result = hooks[0](context)

        assert result == "RESULT"

        context2 = ToolCallHookContext(
            tool_name="other_tool",
            tool_input={},
            tool=mock_tool,
            tool_result="result",
        )
        result2 = hooks[0](context2)

        assert result2 is None


class TestDecoratorAttributes:
    """Test that decorators set proper attributes on functions."""

    def test_before_llm_call_sets_attribute(self):
        """Test that decorator sets is_before_llm_call_hook attribute."""

        @before_llm_call
        def test_hook(context):
            pass

        assert hasattr(test_hook, "is_before_llm_call_hook")
        assert test_hook.is_before_llm_call_hook is True

    def test_before_tool_call_sets_attributes_with_filters(self):
        """Test that decorator with filters sets filter attributes."""

        @before_tool_call(tools=["delete_file"], agents=["Dev"])
        def test_hook(context):
            return None

        assert hasattr(test_hook, "is_before_tool_call_hook")
        assert test_hook.is_before_tool_call_hook is True
        assert hasattr(test_hook, "_filter_tools")
        assert test_hook._filter_tools == ["delete_file"]
        assert hasattr(test_hook, "_filter_agents")
        assert test_hook._filter_agents == ["Dev"]


class TestMultipleDecorators:
    """Test using multiple decorators together."""

    def test_multiple_decorators_all_register(self):
        """Test that multiple decorated functions all register."""

        @before_llm_call
        def hook1(context):
            pass

        @before_llm_call
        def hook2(context):
            pass

        @after_llm_call
        def hook3(context):
            return None

        before_hooks = get_before_llm_call_hooks()
        after_hooks = get_after_llm_call_hooks()

        assert len(before_hooks) == 2
        assert len(after_hooks) == 1

    def test_decorator_and_manual_registration_work_together(self):
        """Test that decorators and manual registration can be mixed."""
        from crewai.hooks import register_before_tool_call_hook

        @before_tool_call
        def decorated_hook(context):
            return None

        def manual_hook(context):
            return None

        register_before_tool_call_hook(manual_hook)

        hooks = get_before_tool_call_hooks()

        assert len(hooks) == 2


class RecordingGuardrailProvider:
    """Small provider fixture that records the latest detached request."""

    def __init__(
        self,
        decision: GuardrailDecision | None = None,
        error: Exception | None = None,
    ) -> None:
        self.decision = (
            decision if decision is not None else GuardrailDecision(allow=True)
        )
        self.error = error
        self.request: GuardrailRequest | None = None

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        self.request = request
        if self.error is not None:
            raise self.error
        return self.decision


class TestGuardrailProviderAdapter:
    """Test the provider adapter built on the existing before-tool hook registry."""

    @staticmethod
    def context() -> ToolCallHookContext:
        tool = Mock()
        tool.name = "Send Email"
        agent = Mock()
        agent.id = "agent-123"
        agent.role = "Support Agent"
        return ToolCallHookContext(
            tool_name="send_email",
            tool_input={"recipient": "qa@example.com", "body": "Hello"},
            tool=tool,
            agent=agent,
        )

    def test_registers_provider_and_allows(self):
        provider = RecordingGuardrailProvider()
        hook = enable_guardrail(provider)

        assert get_before_tool_call_hooks() == [hook]
        assert hook(self.context()) is None
        assert provider.request is not None
        assert provider.request.tool_name == "Send Email"
        assert provider.request.tool_alias == "send_email"
        assert provider.request.agent_id == "agent-123"
        assert provider.request.agent_role == "Support Agent"

    def test_deny_maps_to_false(self):
        provider = RecordingGuardrailProvider(
            GuardrailDecision(allow=False, reason="policy")
        )

        assert enable_guardrail(provider)(self.context()) is False

    def test_provider_receives_detached_tool_input(self):
        context = self.context()
        provider = RecordingGuardrailProvider()

        assert enable_guardrail(provider)(context) is None
        assert provider.request is not None
        provider.request.tool_input["recipient"] = "other@example.com"

        assert context.tool_input["recipient"] == "qa@example.com"

    def test_provider_failure_is_fail_closed_by_default(self):
        provider = RecordingGuardrailProvider(error=RuntimeError("unavailable"))

        assert enable_guardrail(provider)(self.context()) is False

    def test_provider_failure_can_fail_open(self):
        provider = RecordingGuardrailProvider(error=RuntimeError("unavailable"))

        assert enable_guardrail(provider, fail_closed=False)(self.context()) is None

    @pytest.mark.parametrize("fail_closed, expected", [(True, False), (False, None)])
    def test_invalid_provider_result_follows_failure_policy(
        self,
        fail_closed: bool,
        expected: bool | None,
    ):
        provider = RecordingGuardrailProvider()
        provider.decision = object()  # type: ignore[assignment]

        assert enable_guardrail(provider, fail_closed=fail_closed)(self.context()) is expected

    def test_non_boolean_allow_is_rejected(self):
        decision = GuardrailDecision(allow=True)
        object.__setattr__(decision, "allow", 1)
        provider = RecordingGuardrailProvider(decision)

        assert enable_guardrail(provider)(self.context()) is False
