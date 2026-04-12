from __future__ import annotations

from unittest.mock import Mock

from crewai.hooks.guardrails import (
    AllowlistGuardrailProvider,
    GuardrailDecision,
    GuardrailRequest,
    build_guardrail_request,
    disable_guardrail,
    enable_guardrail,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    clear_after_tool_call_hooks,
    clear_before_tool_call_hooks,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
)
import pytest


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    original_before = get_before_tool_call_hooks()
    original_after = get_after_tool_call_hooks()

    clear_before_tool_call_hooks()
    clear_after_tool_call_hooks()

    yield

    clear_before_tool_call_hooks()
    clear_after_tool_call_hooks()

    for hook in original_before:
        register_before_tool_call_hook(hook)
    for hook in original_after:
        register_after_tool_call_hook(hook)


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = Mock()
    tool.name = "dangerous_tool"
    return tool


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.role = "Security Agent"
    return agent


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    task = Mock()
    task.description = "Evaluate the tool call"
    return task


@pytest.fixture
def mock_crew():
    """Create a mock crew for testing."""
    crew = Mock()
    crew.id = "crew-123"
    return crew


def test_build_guardrail_request_uses_normalized_context_snapshot(
    mock_tool, mock_agent, mock_task, mock_crew
):
    """Guardrail requests should expose a normalized snapshot of hook context."""
    context = ToolCallHookContext(
        tool_name="dangerous_tool",
        tool_input={"action": "delete"},
        tool=mock_tool,
        agent=mock_agent,
        task=mock_task,
        crew=mock_crew,
    )

    request = build_guardrail_request(context)
    request.tool_input["action"] = "read"

    assert isinstance(request, GuardrailRequest)
    assert request.tool_name == "dangerous_tool"
    assert request.tool_input == {"action": "read"}
    assert request.agent_role == "Security Agent"
    assert request.task_description == "Evaluate the tool call"
    assert request.crew_id == "crew-123"
    assert context.tool_input == {"action": "delete"}
    assert request.timestamp


def test_enable_guardrail_registers_before_hook():
    """Enabling guardrails should register exactly one before hook."""

    class AllowProvider:
        name = "allow-provider"

        def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
            return GuardrailDecision(allow=True)

    hook = enable_guardrail(AllowProvider())

    hooks = get_before_tool_call_hooks()

    assert len(hooks) == 1
    assert hooks[0] == hook
    disable_guardrail(hook)


def test_disable_guardrail_unregisters_before_hook():
    """Disabling guardrails should remove the registered hook."""

    class AllowProvider:
        name = "allow-provider"

        def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
            return GuardrailDecision(allow=True)

    hook = enable_guardrail(AllowProvider())

    assert disable_guardrail(hook) is True
    assert get_before_tool_call_hooks() == []


def test_enable_guardrail_allows_execution(mock_tool):
    """Allow decisions should not block tool execution."""

    class AllowProvider:
        name = "allow-provider"

        def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
            assert request.tool_name == "dangerous_tool"
            return GuardrailDecision(allow=True)

    hook = enable_guardrail(AllowProvider())
    context = ToolCallHookContext(
        tool_name="dangerous_tool",
        tool_input={"action": "delete"},
        tool=mock_tool,
    )

    result = hook(context)

    assert result is None
    disable_guardrail(hook)


def test_enable_guardrail_denies_execution(mock_tool):
    """Deny decisions should block tool execution."""

    class DenyProvider:
        name = "deny-provider"

        def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
            return GuardrailDecision(allow=False, reason="Shell access is not allowed.")

    hook = enable_guardrail(DenyProvider())
    context = ToolCallHookContext(
        tool_name="dangerous_tool",
        tool_input={"action": "delete"},
        tool=mock_tool,
    )

    result = hook(context)

    assert result is False
    disable_guardrail(hook)


def test_enable_guardrail_uses_request_snapshot(mock_tool):
    """Providers should not be able to mutate live tool inputs by accident."""

    class MutatingProvider:
        name = "mutating-provider"

        def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
            request.tool_input["action"] = "read"
            return GuardrailDecision(allow=True)

    hook = enable_guardrail(MutatingProvider())
    context = ToolCallHookContext(
        tool_name="dangerous_tool",
        tool_input={"action": "delete"},
        tool=mock_tool,
    )

    result = hook(context)

    assert result is None
    assert context.tool_input == {"action": "delete"}
    disable_guardrail(hook)


def test_enable_guardrail_fail_closed_blocks_on_provider_error(mock_tool):
    """Provider errors should block execution when fail_closed is enabled."""

    class FailingProvider:
        name = "failing-provider"

        def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
            raise RuntimeError("Provider unavailable")

    hook = enable_guardrail(FailingProvider(), fail_closed=True)
    context = ToolCallHookContext(
        tool_name="dangerous_tool",
        tool_input={"action": "delete"},
        tool=mock_tool,
    )

    result = hook(context)

    assert result is False
    disable_guardrail(hook)


def test_enable_guardrail_fail_open_allows_on_provider_error(mock_tool):
    """Provider errors should allow execution when fail_closed is disabled."""

    class FailingProvider:
        name = "failing-provider"

        def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
            raise RuntimeError("Provider unavailable")

    hook = enable_guardrail(FailingProvider(), fail_closed=False)
    context = ToolCallHookContext(
        tool_name="dangerous_tool",
        tool_input={"action": "delete"},
        tool=mock_tool,
    )

    result = hook(context)

    assert result is None
    disable_guardrail(hook)


def test_allowlist_guardrail_provider_blocks_denied_tools():
    """The built-in allowlist provider should deny explicitly blocked tools."""
    provider = AllowlistGuardrailProvider(denied_tools=["exec", "delete_file"])

    decision = provider.evaluate(
        GuardrailRequest(
            tool_name="exec",
            tool_input={"command": "rm -rf /"},
        )
    )

    assert decision.allow is False
    assert decision.reason == "'exec' is in the denied tools list."


def test_allowlist_guardrail_provider_allows_allowlisted_tools():
    """The built-in allowlist provider should allow tools inside the allowlist."""
    provider = AllowlistGuardrailProvider(
        allowed_tools=["read_file", "web_search"],
        denied_tools=["exec"],
    )

    decision = provider.evaluate(
        GuardrailRequest(
            tool_name="read_file",
            tool_input={"path": "README.md"},
        )
    )

    assert decision.allow is True
    assert decision.reason is None


def test_allowlist_guardrail_provider_blocks_non_allowlisted_tools():
    """The built-in allowlist provider should fail closed outside the allowlist."""
    provider = AllowlistGuardrailProvider(allowed_tools=["read_file", "web_search"])

    decision = provider.evaluate(
        GuardrailRequest(
            tool_name="exec",
            tool_input={"command": "ls"},
        )
    )

    assert decision.allow is False
    assert decision.reason == "'exec' is not in the allowed tools list."


def test_allowlist_guardrail_provider_allows_tools_with_unsanitized_names():
    """The built-in allowlist provider should allow tools present in the allowlist."""
    provider = AllowlistGuardrailProvider(
        allowed_tools=["Read File", "Web Search"],
    )

    decision = provider.evaluate(
        GuardrailRequest(
            tool_name="read_file",
            tool_input={"path": "notes.txt"},
        )
    )

    assert decision.allow is True
    assert decision.reason is None


def test_allowlist_guardrail_provider_blocks_denied_tools_with_unsanitized_names():
    """Configured deny entries should normalize the same way as tool hooks."""
    provider = AllowlistGuardrailProvider(denied_tools=["Delete File"])

    decision = provider.evaluate(
        GuardrailRequest(
            tool_name="delete_file",
            tool_input={"path": "notes.txt"},
        )
    )

    assert decision.allow is False
    assert decision.reason == "'delete_file' is in the denied tools list."


def test_allowlist_guardrail_provider_prioritizes_denied_tools():
    """Deny rules should win when a tool appears in both lists."""
    provider = AllowlistGuardrailProvider(
        allowed_tools=["exec", "read_file"],
        denied_tools=["exec"],
    )

    decision = provider.evaluate(
        GuardrailRequest(
            tool_name="exec",
            tool_input={"command": "ls"},
        )
    )

    assert decision.allow is False
    assert decision.reason == "'exec' is in the denied tools list."
