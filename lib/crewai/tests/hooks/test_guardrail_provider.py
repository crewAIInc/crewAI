"""Tests for the GuardrailProvider interface and enable_guardrail adapter."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from crewai.hooks.guardrail_provider import (
    GuardrailDecision,
    GuardrailProvider,
    GuardrailRequest,
    _build_guardrail_request,
    enable_guardrail,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_before_tool_call_hooks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    agent.role = "Researcher"
    return agent


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    task = Mock()
    task.description = "Summarize the findings"
    return task


@pytest.fixture
def mock_crew():
    """Create a mock crew for testing."""
    crew = Mock()
    crew.id = "crew-123"
    return crew


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear global hooks before and after each test."""
    from crewai.hooks import tool_hooks

    original_before = tool_hooks._before_tool_call_hooks.copy()
    original_after = tool_hooks._after_tool_call_hooks.copy()

    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()

    yield

    tool_hooks._before_tool_call_hooks.clear()
    tool_hooks._after_tool_call_hooks.clear()
    tool_hooks._before_tool_call_hooks.extend(original_before)
    tool_hooks._after_tool_call_hooks.extend(original_after)


# ---------------------------------------------------------------------------
# Concrete provider used across tests
# ---------------------------------------------------------------------------

class AllowAllProvider:
    """A provider that allows every tool call."""

    name = "allow_all"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return GuardrailDecision(allow=True)

    def health_check(self) -> bool:
        return True


class BlockListProvider:
    """A provider that blocks specific tools by name."""

    name = "block_list"

    def __init__(self, blocked_tools: list[str]) -> None:
        self.blocked_tools = blocked_tools

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name in self.blocked_tools:
            return GuardrailDecision(
                allow=False,
                reason=f"Tool '{request.tool_name}' is blocked by policy",
                metadata={"policy": "block_list"},
            )
        return GuardrailDecision(allow=True)

    def health_check(self) -> bool:
        return True


class ExplodingProvider:
    """A provider that always raises an exception during evaluate."""

    name = "exploding"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        raise RuntimeError("Provider failure!")

    def health_check(self) -> bool:
        return False


class RoleBasedProvider:
    """A provider that restricts tool access based on agent role."""

    name = "role_based"

    def __init__(self, permissions: dict[str, list[str]]) -> None:
        self.permissions = permissions

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        role = request.agent_role
        if role is None:
            return GuardrailDecision(allow=True)
        allowed = self.permissions.get(role)
        if allowed is not None and request.tool_name not in allowed:
            return GuardrailDecision(
                allow=False,
                reason=f"Agent '{role}' is not permitted to use '{request.tool_name}'",
            )
        return GuardrailDecision(allow=True)

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# GuardrailRequest tests
# ---------------------------------------------------------------------------

class TestGuardrailRequest:
    """Test GuardrailRequest construction and defaults."""

    def test_required_fields(self):
        req = GuardrailRequest(tool_name="search", tool_input={"q": "hello"})
        assert req.tool_name == "search"
        assert req.tool_input == {"q": "hello"}

    def test_optional_fields_default_to_none_or_empty(self):
        req = GuardrailRequest(tool_name="search", tool_input={})
        assert req.agent_role is None
        assert req.task_description is None
        assert req.crew_id is None
        assert req.timestamp == ""

    def test_all_fields_populated(self):
        req = GuardrailRequest(
            tool_name="write_file",
            tool_input={"path": "/tmp/x"},
            agent_role="Developer",
            task_description="Write config",
            crew_id="crew-42",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert req.tool_name == "write_file"
        assert req.tool_input == {"path": "/tmp/x"}
        assert req.agent_role == "Developer"
        assert req.task_description == "Write config"
        assert req.crew_id == "crew-42"
        assert req.timestamp == "2025-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# GuardrailDecision tests
# ---------------------------------------------------------------------------

class TestGuardrailDecision:
    """Test GuardrailDecision construction and defaults."""

    def test_allow_decision(self):
        dec = GuardrailDecision(allow=True)
        assert dec.allow is True
        assert dec.reason is None
        assert dec.metadata == {}

    def test_deny_decision_with_reason(self):
        dec = GuardrailDecision(allow=False, reason="Blocked by policy")
        assert dec.allow is False
        assert dec.reason == "Blocked by policy"

    def test_decision_with_metadata(self):
        dec = GuardrailDecision(
            allow=False,
            reason="Denied",
            metadata={"policy_id": "P-001", "audit": True},
        )
        assert dec.metadata == {"policy_id": "P-001", "audit": True}


# ---------------------------------------------------------------------------
# GuardrailProvider protocol tests
# ---------------------------------------------------------------------------

class TestGuardrailProviderProtocol:
    """Test that the runtime_checkable protocol works correctly."""

    def test_allow_all_provider_is_guardrail_provider(self):
        assert isinstance(AllowAllProvider(), GuardrailProvider)

    def test_block_list_provider_is_guardrail_provider(self):
        assert isinstance(BlockListProvider(blocked_tools=[]), GuardrailProvider)

    def test_exploding_provider_is_guardrail_provider(self):
        assert isinstance(ExplodingProvider(), GuardrailProvider)

    def test_role_based_provider_is_guardrail_provider(self):
        assert isinstance(
            RoleBasedProvider(permissions={}), GuardrailProvider
        )

    def test_plain_object_is_not_guardrail_provider(self):
        """An object without evaluate/health_check is not a GuardrailProvider."""
        assert not isinstance(object(), GuardrailProvider)

    def test_partial_implementation_is_not_guardrail_provider(self):
        """An object with only evaluate but no name/health_check is not a provider."""

        class Incomplete:
            def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
                return GuardrailDecision(allow=True)

        assert not isinstance(Incomplete(), GuardrailProvider)


# ---------------------------------------------------------------------------
# _build_guardrail_request tests
# ---------------------------------------------------------------------------

class TestBuildGuardrailRequest:
    """Test the internal helper that converts ToolCallHookContext to GuardrailRequest."""

    def test_full_context(self, mock_tool, mock_agent, mock_task, mock_crew):
        context = ToolCallHookContext(
            tool_name="search",
            tool_input={"query": "AI"},
            tool=mock_tool,
            agent=mock_agent,
            task=mock_task,
            crew=mock_crew,
        )
        req = _build_guardrail_request(context)

        assert req.tool_name == "search"
        assert req.tool_input == {"query": "AI"}
        assert req.agent_role == "Researcher"
        assert req.task_description == "Summarize the findings"
        assert req.crew_id == "crew-123"
        assert req.timestamp != ""  # should be populated

    def test_minimal_context(self, mock_tool):
        context = ToolCallHookContext(
            tool_name="noop",
            tool_input={},
            tool=mock_tool,
        )
        req = _build_guardrail_request(context)

        assert req.tool_name == "noop"
        assert req.tool_input == {}
        assert req.agent_role is None
        assert req.task_description is None
        assert req.crew_id is None
        assert req.timestamp != ""

    def test_agent_without_role_attribute(self, mock_tool):
        """Agent-like objects without a role attribute should yield None."""
        agent_no_role = Mock(spec=[])  # no attributes at all
        context = ToolCallHookContext(
            tool_name="tool",
            tool_input={},
            tool=mock_tool,
            agent=agent_no_role,
        )
        req = _build_guardrail_request(context)
        assert req.agent_role is None


# ---------------------------------------------------------------------------
# enable_guardrail tests
# ---------------------------------------------------------------------------

class TestEnableGuardrail:
    """Test the enable_guardrail adapter function."""

    def test_enable_registers_a_before_hook(self):
        provider = AllowAllProvider()
        disable = enable_guardrail(provider)

        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 1

        disable()

    def test_disable_removes_the_hook(self):
        provider = AllowAllProvider()
        disable = enable_guardrail(provider)

        assert len(get_before_tool_call_hooks()) == 1

        result = disable()
        assert result is True
        assert len(get_before_tool_call_hooks()) == 0

    def test_disable_returns_false_when_already_removed(self):
        provider = AllowAllProvider()
        disable = enable_guardrail(provider)

        disable()  # first removal
        result = disable()  # second removal – already gone
        assert result is False

    def test_allow_all_provider_permits_tool_call(self, mock_tool):
        provider = AllowAllProvider()
        disable = enable_guardrail(provider)

        context = ToolCallHookContext(
            tool_name="any_tool",
            tool_input={"x": 1},
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        result = hooks[0](context)
        assert result is None  # None means allow

        disable()

    def test_block_list_provider_denies_blocked_tool(self, mock_tool):
        provider = BlockListProvider(blocked_tools=["ShellTool"])
        disable = enable_guardrail(provider)

        context = ToolCallHookContext(
            tool_name="ShellTool",
            tool_input={"cmd": "rm -rf /"},
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        result = hooks[0](context)
        assert result is False  # blocked

        disable()

    def test_block_list_provider_allows_non_blocked_tool(self, mock_tool):
        provider = BlockListProvider(blocked_tools=["ShellTool"])
        disable = enable_guardrail(provider)

        context = ToolCallHookContext(
            tool_name="SearchTool",
            tool_input={"q": "hello"},
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        result = hooks[0](context)
        assert result is None  # allowed

        disable()

    def test_role_based_provider_blocks_unauthorized_agent(
        self, mock_tool, mock_agent
    ):
        provider = RoleBasedProvider(
            permissions={"Researcher": ["SearchTool", "ReadFileTool"]}
        )
        disable = enable_guardrail(provider)

        context = ToolCallHookContext(
            tool_name="ShellTool",
            tool_input={},
            tool=mock_tool,
            agent=mock_agent,  # role = "Researcher"
        )

        hooks = get_before_tool_call_hooks()
        result = hooks[0](context)
        assert result is False  # Researcher can't use ShellTool

        disable()

    def test_role_based_provider_allows_authorized_agent(
        self, mock_tool, mock_agent
    ):
        provider = RoleBasedProvider(
            permissions={"Researcher": ["SearchTool"]}
        )
        disable = enable_guardrail(provider)

        context = ToolCallHookContext(
            tool_name="SearchTool",
            tool_input={},
            tool=mock_tool,
            agent=mock_agent,
        )

        hooks = get_before_tool_call_hooks()
        result = hooks[0](context)
        assert result is None  # allowed

        disable()

    def test_fail_closed_blocks_on_exception(self, mock_tool):
        """When fail_closed=True (default), provider exceptions block the tool."""
        provider = ExplodingProvider()
        disable = enable_guardrail(provider, fail_closed=True)

        context = ToolCallHookContext(
            tool_name="any_tool",
            tool_input={},
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        result = hooks[0](context)
        assert result is False  # blocked due to exception

        disable()

    def test_fail_open_allows_on_exception(self, mock_tool):
        """When fail_closed=False, provider exceptions allow the tool."""
        provider = ExplodingProvider()
        disable = enable_guardrail(provider, fail_closed=False)

        context = ToolCallHookContext(
            tool_name="any_tool",
            tool_input={},
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        result = hooks[0](context)
        assert result is None  # allowed despite exception

        disable()

    def test_multiple_providers_all_must_allow(self, mock_tool):
        """When multiple providers are enabled, all must allow for the tool to proceed."""
        provider1 = AllowAllProvider()
        provider2 = BlockListProvider(blocked_tools=["DangerousTool"])

        disable1 = enable_guardrail(provider1)
        disable2 = enable_guardrail(provider2)

        hooks = get_before_tool_call_hooks()
        assert len(hooks) == 2

        # Safe tool – both allow
        context_safe = ToolCallHookContext(
            tool_name="SafeTool",
            tool_input={},
            tool=mock_tool,
        )
        results = [h(context_safe) for h in hooks]
        assert all(r is None for r in results)

        # Dangerous tool – first allows, second blocks
        context_danger = ToolCallHookContext(
            tool_name="DangerousTool",
            tool_input={},
            tool=mock_tool,
        )
        blocked = False
        for hook in hooks:
            result = hook(context_danger)
            if result is False:
                blocked = True
                break
        assert blocked is True

        disable1()
        disable2()

    def test_guardrail_request_timestamp_is_set(self, mock_tool):
        """The hook should populate the timestamp in the GuardrailRequest."""
        received_requests: list[GuardrailRequest] = []

        class SpyProvider:
            name = "spy"

            def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
                received_requests.append(request)
                return GuardrailDecision(allow=True)

            def health_check(self) -> bool:
                return True

        provider = SpyProvider()
        disable = enable_guardrail(provider)

        context = ToolCallHookContext(
            tool_name="tool",
            tool_input={"key": "val"},
            tool=mock_tool,
        )

        hooks = get_before_tool_call_hooks()
        hooks[0](context)

        assert len(received_requests) == 1
        assert received_requests[0].timestamp != ""
        # Should be a valid ISO 8601 string
        assert "T" in received_requests[0].timestamp

        disable()

    def test_guardrail_context_fields_passed_through(
        self, mock_tool, mock_agent, mock_task, mock_crew
    ):
        """Verify that agent_role, task_description, crew_id are forwarded."""
        received_requests: list[GuardrailRequest] = []

        class SpyProvider:
            name = "spy"

            def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
                received_requests.append(request)
                return GuardrailDecision(allow=True)

            def health_check(self) -> bool:
                return True

        provider = SpyProvider()
        disable = enable_guardrail(provider)

        context = ToolCallHookContext(
            tool_name="search",
            tool_input={"q": "test"},
            tool=mock_tool,
            agent=mock_agent,
            task=mock_task,
            crew=mock_crew,
        )

        hooks = get_before_tool_call_hooks()
        hooks[0](context)

        req = received_requests[0]
        assert req.tool_name == "search"
        assert req.tool_input == {"q": "test"}
        assert req.agent_role == "Researcher"
        assert req.task_description == "Summarize the findings"
        assert req.crew_id == "crew-123"

        disable()

    def test_decision_metadata_is_accessible(self, mock_tool):
        """Provider metadata in the decision can be used for auditing."""

        class AuditProvider:
            name = "audit"

            def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
                return GuardrailDecision(
                    allow=True,
                    metadata={"trace_id": "abc-123", "evaluated_at": request.timestamp},
                )

            def health_check(self) -> bool:
                return True

        provider = AuditProvider()
        # Just verify the provider works; metadata is returned but
        # not directly exposed by the hook (it's for provider-side use)
        req = GuardrailRequest(tool_name="tool", tool_input={})
        decision = provider.evaluate(req)
        assert decision.metadata["trace_id"] == "abc-123"
