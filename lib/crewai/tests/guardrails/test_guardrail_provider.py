"""
Tests for CorrectoverGuardrailProvider.

Verifies that the reference implementation correctly implements the
GuardrailProvider protocol from crewAIInc/crewAI#4877.
"""

from __future__ import annotations

import time
import pytest

from crewai.guardrails.providers import (
    CorrectoverGuardrailProvider,
    GuardrailDecision,
    GuardrailProvider,
    GuardrailRequest,
    DimensionResult,
    VerificationReport,
)
from crewai.guardrails.enable import ToolCallContext, enable_guardrail


def _make_request(**kwargs) -> GuardrailRequest:
    """Helper to create a GuardrailRequest with defaults."""
    defaults = {
        "tool_name": "search",
        "tool_input": {"query": "test"},
        "timestamp": time.time(),
    }
    defaults.update(kwargs)
    return GuardrailRequest(**defaults)


def _make_context(**kwargs) -> ToolCallContext:
    """Helper to create a ToolCallContext with defaults."""
    defaults = {
        "tool_name": "search",
        "tool_input": {"query": "test"},
    }
    defaults.update(kwargs)
    return ToolCallContext(**defaults)


class TestProviderProtocol:
    """Verify the provider implements the GuardrailProvider protocol."""

    def test_has_name_attribute(self):
        provider = CorrectoverGuardrailProvider()
        assert provider.name == "correctover"

    def test_has_evaluate_method(self):
        provider = CorrectoverGuardrailProvider()
        assert hasattr(provider, "evaluate")
        assert callable(provider.evaluate)

    def test_is_runtime_checkable(self):
        provider = CorrectoverGuardrailProvider()
        assert isinstance(provider, GuardrailProvider)


class TestAllowDecisions:
    """Verify correct allow decisions."""

    def test_valid_request_allowed(self):
        provider = CorrectoverGuardrailProvider()
        request = _make_request()
        decision = provider.evaluate(request)
        assert decision.allow is True
        assert isinstance(decision.reason, str)

    def test_decision_is_guardrail_decision(self):
        provider = CorrectoverGuardrailProvider()
        request = _make_request()
        decision = provider.evaluate(request)
        assert isinstance(decision, GuardrailDecision)

    def test_metadata_present(self):
        provider = CorrectoverGuardrailProvider()
        request = _make_request()
        decision = provider.evaluate(request)
        assert decision.metadata is not None
        assert "dimensions_checked" in decision.metadata

    def test_action_id_present(self):
        """Verify decision includes action_id for audit traceability."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request()
        decision = provider.evaluate(request)
        assert decision.action_id is not None
        assert isinstance(decision.action_id, str)
        assert len(decision.action_id) == 16  # sha256 hex truncated to 16 chars

    def test_action_id_deterministic(self):
        """Same request produces same action_id."""
        provider = CorrectoverGuardrailProvider()
        ts = time.time()
        request = _make_request(timestamp=ts)
        d1 = provider.evaluate(request)
        d2 = provider.evaluate(request)
        assert d1.action_id == d2.action_id

    def test_action_id_deterministic_without_timestamp(self):
        """action_id is deterministic even when timestamp is None."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(timestamp=None)
        d1 = provider.evaluate(request)
        d2 = provider.evaluate(request)
        assert d1.action_id == d2.action_id


class TestDenyDecisions:
    """Verify correct deny decisions."""

    def test_blocked_tool_denied(self):
        provider = CorrectoverGuardrailProvider(blocked_tools={"dangerous_tool"})
        request = _make_request(tool_name="dangerous_tool")
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert decision.metadata["dimension_details"]["schema"]["detail"] is not None
        assert "blocked" in decision.metadata["dimension_details"]["schema"]["detail"].lower()
        assert decision.action_id is not None

    def test_non_whitelisted_tool_denied(self):
        provider = CorrectoverGuardrailProvider(allowed_tools={"only_this"})
        request = _make_request(tool_name="other_tool")
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "allowed list" in decision.metadata["dimension_details"]["schema"]["detail"].lower()

    def test_missing_agent_id_denied(self):
        provider = CorrectoverGuardrailProvider(require_agent_identity=True)
        request = _make_request(agent_id=None)
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "agent_id" in decision.metadata["dimension_details"]["identity"]["detail"].lower()

    def test_unauthorized_agent_denied(self):
        provider = CorrectoverGuardrailProvider(allowed_agents={"agent-123"})
        request = _make_request(agent_id="agent-999")
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "not in allowed list" in decision.metadata["dimension_details"]["identity"]["detail"].lower()

    def test_empty_tool_name_denied(self):
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_name="")
        decision = provider.evaluate(request)
        assert decision.allow is False

    def test_empty_tool_input_allowed(self):
        """Empty tool_input is structurally valid (not the same as None)."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_input={})
        decision = provider.evaluate(request)
        assert decision.allow is True


class TestFailClosed:
    """Verify fail-closed behavior."""

    def test_fail_closed_default(self):
        provider = CorrectoverGuardrailProvider()
        assert provider.fail_closed is True

    def test_fail_closed_false_on_error(self):
        """With fail_closed=False, provider errors should allow execution."""
        provider = CorrectoverGuardrailProvider(fail_closed=False)
        assert provider.fail_closed is False


class TestTimestampField:
    """Verify timestamp field in GuardrailRequest."""

    def test_request_has_timestamp(self):
        request = _make_request()
        assert request.timestamp is not None
        assert isinstance(request.timestamp, float)

    def test_request_timestamp_optional(self):
        request = GuardrailRequest(
            tool_name="test",
            tool_input={},
        )
        assert request.timestamp is None

    def test_latency_check_uses_timestamp(self):
        """Verify latency check uses request timestamp for actual measurement."""
        provider = CorrectoverGuardrailProvider(max_latency_ms=0.001)
        request = _make_request(timestamp=time.time() - 1.0)
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "latency" in decision.reason.lower()


class TestCostPolicy:
    """Verify cost policy enforcement."""

    def test_negative_cost_threshold_rejected(self):
        provider = CorrectoverGuardrailProvider(max_cost_usd=-1.0)
        request = _make_request()
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "cost" in decision.reason.lower()

    def test_valid_cost_threshold_allowed(self):
        provider = CorrectoverGuardrailProvider(max_cost_usd=10.0)
        request = _make_request()
        decision = provider.evaluate(request)
        assert decision.allow is True


class TestLatencyPolicy:
    """Verify latency policy enforcement."""

    def test_negative_latency_threshold_rejected(self):
        provider = CorrectoverGuardrailProvider(max_latency_ms=-100)
        request = _make_request()
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "latency" in decision.reason.lower()

    def test_valid_latency_threshold_allowed(self):
        provider = CorrectoverGuardrailProvider(max_latency_ms=5000)
        request = _make_request()
        decision = provider.evaluate(request)
        assert decision.allow is True


class TestEnableGuardrail:
    """Verify the enable_guardrail integration."""

    def test_returns_hook_function(self):
        provider = CorrectoverGuardrailProvider()
        hook = enable_guardrail(provider)
        assert callable(hook)

    def test_allow_returns_none(self):
        """When allowed, hook returns None (pass-through)."""
        provider = CorrectoverGuardrailProvider()
        hook = enable_guardrail(provider)
        context = _make_context()
        result = hook(context)
        assert result is None

    def test_deny_returns_false(self):
        """When denied, hook returns False (block execution)."""
        provider = CorrectoverGuardrailProvider(blocked_tools={"search"})
        hook = enable_guardrail(provider)
        context = _make_context(tool_name="search")
        result = hook(context)
        assert result is False

    def test_invalid_decision_fails_closed(self):
        """Provider errors block execution by default."""

        class BrokenProvider:
            name = "broken"

            def evaluate(self, request):
                raise RuntimeError("provider crashed")

        hook = enable_guardrail(BrokenProvider(), fail_closed=True)
        context = _make_context()
        result = hook(context)
        assert result is False

    def test_invalid_decision_fails_open(self):
        """Provider errors can pass through if fail_closed=False."""

        class BrokenProvider:
            name = "broken"

            def evaluate(self, request):
                raise RuntimeError("provider crashed")

        hook = enable_guardrail(BrokenProvider(), fail_closed=False)
        context = _make_context()
        result = hook(context)
        assert result is None

    def test_context_timestamp_preserved(self):
        """Context timestamp is preserved in the request for latency checks."""
        provider = CorrectoverGuardrailProvider(max_latency_ms=100)
        hook = enable_guardrail(provider)
        # Use a very old timestamp to trigger latency failure
        context = _make_context(timestamp=time.time() - 10.0)
        result = hook(context)
        assert result is False

    def test_context_no_timestamp_uses_current_time(self):
        """When context has no timestamp, current time is used."""
        provider = CorrectoverGuardrailProvider()
        hook = enable_guardrail(provider)
        context = _make_context()
        result = hook(context)
        assert result is None


class TestRequestStats:
    """Verify provider statistics."""

    def test_initial_stats(self):
        provider = CorrectoverGuardrailProvider()
        stats = provider.get_stats()
        assert stats["total_requests_evaluated"] == 0
        assert stats["name"] == "correctover"

    def test_stats_after_evaluation(self):
        provider = CorrectoverGuardrailProvider()
        request = _make_request()
        provider.evaluate(request)
        provider.evaluate(request)
        stats = provider.get_stats()
        assert stats["total_requests_evaluated"] == 2

    def test_stats_include_configuration(self):
        provider = CorrectoverGuardrailProvider(
            blocked_tools={"x"},
            max_cost_usd=5.0,
        )
        stats = provider.get_stats()
        config = stats["configuration"]
        assert config["blocked_tools"] == ["x"]
        assert config["max_cost_usd"] == 5.0


class TestVerificationReport:
    """Verify VerificationReport.failed_dimensions returns strings."""

    def test_failed_dimensions_returns_strings(self):
        report = VerificationReport(
            dimensions=[
                DimensionResult("structure", True),
                DimensionResult("schema", False, "blocked"),
                DimensionResult("cost", False, "exceeded"),
            ],
            allow=False,
        )
        failed = report.failed_dimensions
        assert all(isinstance(d, str) for d in failed)
        assert failed == ["schema", "cost"]

    def test_failed_dimensions_empty_when_all_pass(self):
        report = VerificationReport(
            dimensions=[
                DimensionResult("structure", True),
                DimensionResult("schema", True),
            ],
            allow=True,
        )
        assert report.failed_dimensions == []


class TestConformanceVector:
    """
    Conformance vector for the GuardrailProvider contract.
    Verifies structural compatibility with the upstream patch kit.
    """

    def test_request_fields_match_upstream(self):
        """GuardrailRequest has all fields from the upstream contract."""
        ts = time.time()
        request = _make_request(
            agent_id="agent-1",
            agent_role="researcher",
            task_description="search for info",
            crew_id="crew-1",
            timestamp=ts,
        )
        assert request.tool_name == "search"
        assert request.tool_input == {"query": "test"}
        assert request.agent_id == "agent-1"
        assert request.agent_role == "researcher"
        assert request.task_description == "search for info"
        assert request.crew_id == "crew-1"
        assert request.timestamp == ts

    def test_decision_fields_match_upstream(self):
        """GuardrailDecision has all fields from the upstream contract including action_id."""
        decision = GuardrailDecision(
            allow=True,
            reason="all checks passed",
            metadata={"key": "value"},
            action_id="abc123def456",
        )
        assert decision.allow is True
        assert decision.reason == "all checks passed"
        assert decision.metadata == {"key": "value"}
        assert decision.action_id == "abc123def456"

    def test_decision_is_frozen(self):
        """GuardrailDecision is immutable (frozen dataclass)."""
        decision = GuardrailDecision(allow=True)
        with pytest.raises(AttributeError):
            decision.allow = False

    def test_request_is_frozen(self):
        """GuardrailRequest is immutable (frozen dataclass)."""
        request = _make_request()
        with pytest.raises(AttributeError):
            request.tool_name = "other"

    def test_provider_name_is_string(self):
        provider = CorrectoverGuardrailProvider()
        assert isinstance(provider.name, str)
        assert provider.name == "correctover"

    def test_evaluate_returns_guardrail_decision(self):
        provider = CorrectoverGuardrailProvider()
        request = _make_request()
        decision = provider.evaluate(request)
        assert isinstance(decision, GuardrailDecision)
        assert isinstance(decision.allow, bool)


class TestMalformedRequestSafety:
    """
    Security: malformed requests must NEVER escape evaluate().
    They stay on the deny path (fail_closed=True) or structured allow path
    (fail_closed=False) — never raise unhandled exceptions.
    """

    def test_non_string_tool_name_denied(self):
        """Non-string tool_name is denied, not crashed."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_name=123)
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "structure" in decision.reason.lower()

    def test_non_string_tool_name_fail_closed_false(self):
        """Non-string tool_name with fail_closed=False returns structured allow."""
        provider = CorrectoverGuardrailProvider(fail_closed=False)
        request = _make_request(tool_name=123)
        decision = provider.evaluate(request)
        # Structure check catches it → deny (even with fail_closed=False,
        # because this is a dimension failure, not a provider error)
        assert decision.allow is False
        assert isinstance(decision, GuardrailDecision)

    def test_non_mapping_tool_input_denied(self):
        """Non-Mapping tool_input is denied via structure check."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_input="not a dict")
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "structure" in decision.reason.lower()

    def test_none_tool_input_denied(self):
        """None tool_input is denied via structure check."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_input=None)
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "structure" in decision.reason.lower()

    def test_list_tool_input_denied(self):
        """List tool_input is denied (not a Mapping)."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_input=[1, 2, 3])
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "structure" in decision.reason.lower()

    def test_non_string_agent_id_denied(self):
        """Non-string agent_id is denied via identity check (type guard at top level)."""
        provider = CorrectoverGuardrailProvider(
            require_agent_identity=True,
        )
        request = _make_request(agent_id=42)
        decision = provider.evaluate(request)
        # agent_id=42 is truthy but not a string → identity check catches it
        # regardless of whether allowed_agents is set
        assert decision.allow is False
        assert "identity" in decision.reason.lower()

    def test_mixed_key_tool_input_denied(self):
        """tool_input with mixed key types is denied (json.dumps sort_keys fails on int+str)."""
        provider = CorrectoverGuardrailProvider()
        # Note: GuardrailRequest accepts any Mapping, including dict with int keys
        request = GuardrailRequest(
            tool_name="test",
            tool_input={1: "value", "key": "other"},
        )
        decision = provider.evaluate(request)
        # json.dumps(sort_keys=True) raises TypeError on mixed int/str keys,
        # caught by evaluate()'s fail_closed safety net → deny
        assert decision.allow is False
        assert isinstance(decision, GuardrailDecision)

    def test_allowed_agents_with_missing_agent_id_denied(self):
        """When allowed_agents is configured, requests with agent_id=None must be denied."""
        provider = CorrectoverGuardrailProvider(
            allowed_agents={"agent-1", "agent-2"},
            require_agent_identity=False,  # even without require_agent_identity
        )
        request = _make_request(agent_id=None)
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert "identity" in decision.reason.lower()
        detail = decision.metadata["dimension_details"]["identity"]["detail"]
        assert "required" in detail.lower()

    def test_action_id_generated_for_malformed_input(self):
        """action_id is still generated even with non-string tool_name."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_name=123)
        decision = provider.evaluate(request)
        # action_id is generated because _generate_action_id handles non-str
        assert decision.action_id is not None
        assert isinstance(decision.action_id, str)
        assert len(decision.action_id) == 16

    def test_action_id_deterministic_for_malformed_input(self):
        """action_id is deterministic even with malformed inputs."""
        provider = CorrectoverGuardrailProvider()
        request = _make_request(tool_name=456, timestamp=None)
        d1 = provider.evaluate(request)
        d2 = provider.evaluate(request)
        assert d1.action_id == d2.action_id

    def test_malformed_request_never_raises(self):
        """No malformed input combination should raise an unhandled exception."""
        provider = CorrectoverGuardrailProvider(
            blocked_tools={"x"},
            allowed_agents={"agent-1"},
            require_agent_identity=True,
        )
        # Worst-case: everything is wrong
        bad_inputs = [
            {"tool_name": None, "tool_input": None},
            {"tool_name": 123, "tool_input": 456},
            {"tool_name": [1, 2], "tool_input": {"key": "val"}},
            {"tool_name": "", "tool_input": None},
            {"tool_name": 0, "tool_input": []},
            {"tool_name": "x", "tool_input": {1: 2}},
        ]
        for bad in bad_inputs:
            request = GuardrailRequest(**bad)
            # This must NEVER raise
            decision = provider.evaluate(request)
            assert isinstance(decision, GuardrailDecision)
            assert isinstance(decision.allow, bool)
