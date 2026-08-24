# Test suite for CCS-conformant GuardrailProvider implementation.
# CCS — Correctover Conformance Shape
# Reference: https://datatracker.ietf.org/doc/draft-correctover-ccs/

"""
Comprehensive test suite for GuardrailProvider.

Coverage:
- GuardrailDecisionV1: frozen, expiry, integrity verification
- ActionEnvelopeV1: digest, frozen, duration
- compute_decision_id: deterministic, content-addressed, expires_at boundary, key order
- All 4 provider implementations
- Custom provider extensibility
- AuditTrail CRUD
- GuardrailContext: allow/block/decision recording/on_deny callback/integrity
- make_guardrail_hook: callable, stash context, accumulate
- detect_missing_guardrail: corresponds to AS-GUARDRAIL-MISS-001
- 3 real-world scenarios end-to-end
"""

import time
import hashlib
import json
import pytest
from dataclasses import FrozenInstanceError

from crewai.guardrails.guardrail_provider import (
    GuardrailDecisionV1,
    ActionEnvelopeV1,
    GuardrailProvider,
    AllowAllGuardrailProvider,
    DenyAllGuardrailProvider,
    ToolListGuardrailProvider,
    CKGGuardrailProvider,
    AuditTrail,
    ToolCallHookContext,
    GuardrailContext,
    make_guardrail_hook,
    detect_missing_guardrail,
    compute_decision_id,
    canonical_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAgent:
    def __init__(self, role: str = "researcher"):
        self.role = role


def make_context(tool_name="read_file", tool_input=None, agent_role="researcher"):
    return ToolCallHookContext(
        tool_name=tool_name,
        tool_input=tool_input or {},
        agent=MockAgent(role=agent_role),
    )


# ===========================================================================
# TestGuardrailDecisionV1
# ===========================================================================

class TestGuardrailDecisionV1:
    def test_frozen(self):
        claims = {"tool_name": "read_file", "agent_role": "researcher"}
        did = compute_decision_id(claims)
        d = GuardrailDecisionV1(decision_id=did, authorized=True, claims=claims)
        with pytest.raises(FrozenInstanceError):
            d.authorized = False

    def test_expiry_not_expired(self):
        claims = {"test": True}
        did = compute_decision_id(claims, expires_at=time.time() + 3600)
        d = GuardrailDecisionV1(
            decision_id=did, authorized=True, claims=claims,
            expires_at=time.time() + 3600,
        )
        assert not d.is_expired()

    def test_expiry_expired(self):
        claims = {"test": True}
        past = time.time() - 10
        did = compute_decision_id(claims, expires_at=past)
        d = GuardrailDecisionV1(
            decision_id=did, authorized=True, claims=claims, expires_at=past,
        )
        assert d.is_expired()

    def test_integrity_pass(self):
        claims = {"tool_name": "shell", "agent_role": "coder"}
        did = compute_decision_id(claims)
        d = GuardrailDecisionV1(decision_id=did, authorized=True, claims=claims)
        assert d.verify_integrity() is True

    def test_integrity_fail_tampered_claims(self):
        claims = {"tool_name": "shell"}
        did = compute_decision_id(claims)
        # Tamper with claims
        tampered_claims = {"tool_name": "rm -rf /"}
        d = GuardrailDecisionV1(decision_id=did, authorized=True, claims=tampered_claims)
        assert d.verify_integrity() is False

    def test_integrity_fail_tampered_expiry(self):
        claims = {"tool_name": "shell"}
        expires = time.time() + 3600
        did = compute_decision_id(claims, expires_at=expires)
        # Use correct decision_id but wrong expires_at
        d = GuardrailDecisionV1(
            decision_id=did, authorized=True, claims=claims,
            expires_at=time.time() + 9999,
        )
        assert d.verify_integrity() is False

    def test_no_expiry(self):
        claims = {"test": 1}
        d = GuardrailDecisionV1(
            decision_id=compute_decision_id(claims),
            authorized=True, claims=claims, expires_at=None,
        )
        assert not d.is_expired()
        assert d.verify_integrity() is True

    def test_to_dict(self):
        claims = {"a": 1}
        did = compute_decision_id(claims)
        d = GuardrailDecisionV1(decision_id=did, authorized=True, claims=claims)
        result = d.to_dict()
        assert result["decision_id"] == did
        assert result["authorized"] is True
        assert result["claims"] == claims
        assert result["expires_at"] is None


# ===========================================================================
# TestActionEnvelopeV1
# ===========================================================================

class TestActionEnvelopeV1:
    def test_digest_result_none(self):
        assert ActionEnvelopeV1.digest_result(None) == hashlib.sha256(b"").hexdigest()

    def test_digest_result_empty_string(self):
        assert ActionEnvelopeV1.digest_result("") == hashlib.sha256(b"").hexdigest()

    def test_digest_result_differs(self):
        d1 = ActionEnvelopeV1.digest_result({"output": "hello"})
        d2 = ActionEnvelopeV1.digest_result({"output": "world"})
        assert d1 != d2

    def test_digest_result_deterministic(self):
        r = {"key": "value", "num": 42}
        assert ActionEnvelopeV1.digest_result(r) == ActionEnvelopeV1.digest_result(r)

    def test_envelope_frozen(self):
        e = ActionEnvelopeV1(
            decision_id="abc",
            tool_result_digest="def",
            executed_at=time.time(),
            duration_ms=10.5,
        )
        with pytest.raises(FrozenInstanceError):
            e.duration_ms = 99

    def test_envelope_no_raw_result(self):
        """Envelope only stores digest, never raw result."""
        e = ActionEnvelopeV1(
            decision_id="abc",
            tool_result_digest=ActionEnvelopeV1.digest_result({"secret": "data"}),
            executed_at=time.time(),
            duration_ms=5.0,
        )
        d = e.to_dict()
        assert "tool_result_digest" in d
        assert "secret" not in str(d)  # raw value not leaked


# ===========================================================================
# TestComputeDecisionId
# ===========================================================================

class TestComputeDecisionId:
    def test_deterministic(self):
        claims = {"a": 1, "b": 2}
        assert compute_decision_id(claims) == compute_decision_id(claims)

    def test_content_addressed(self):
        c1 = {"a": 1, "b": 2}
        c2 = {"a": 1, "b": 3}
        assert compute_decision_id(c1) != compute_decision_id(c2)

    def test_expires_at_in_preimage(self):
        """Same claims with different expires_at must yield different ids."""
        claims = {"x": 1}
        id1 = compute_decision_id(claims, expires_at=1000.0)
        id2 = compute_decision_id(claims, expires_at=2000.0)
        assert id1 != id2

    def test_expires_at_none_boundary(self):
        """
        When expires_at is None, preimage is claims alone.
        Must differ from any case where expires_at is set.
        """
        claims = {"x": 1}
        id_none = compute_decision_id(claims, expires_at=None)
        id_with = compute_decision_id(claims, expires_at=1000.0)
        assert id_none != id_with

        # Verify the None case doesn't include _expires_at key in serialization
        preimage_none = claims.copy()
        expected = hashlib.sha256(
            canonical_json(preimage_none).encode("utf-8")
        ).hexdigest()
        assert id_none == expected

    def test_key_order_independent(self):
        c1 = {"z": 26, "a": 1, "m": 13}
        c2 = {"a": 1, "m": 13, "z": 26}
        assert compute_decision_id(c1) == compute_decision_id(c2)

    def test_hex_format(self):
        did = compute_decision_id({"test": True})
        assert len(did) == 64
        assert all(c in "0123456789abcdef" for c in did)


# ===========================================================================
# TestAllowAllGuardrailProvider
# ===========================================================================

class TestAllowAll:
    def test_allows_any_tool(self):
        p = AllowAllGuardrailProvider()
        ctx = make_context(tool_name="dangerous_tool")
        d = p.authorize(ctx)
        assert d.authorized is True
        assert d.verify_integrity()

    def test_claims_contain_context(self):
        p = AllowAllGuardrailProvider()
        ctx = make_context(tool_name="my_tool", agent_role="manager")
        d = p.authorize(ctx)
        assert d.claims["tool_name"] == "my_tool"
        assert d.claims["agent_role"] == "manager"


# ===========================================================================
# TestDenyAllGuardrailProvider
# ===========================================================================

class TestDenyAll:
    def test_denies_everything(self):
        p = DenyAllGuardrailProvider()
        ctx = make_context()
        d = p.authorize(ctx)
        assert d.authorized is False
        assert d.verify_integrity()


# ===========================================================================
# TestToolListGuardrailProvider
# ===========================================================================

class TestToolList:
    def test_allowlist_pass(self):
        p = ToolListGuardrailProvider(allowed_tools={"read_file", "search_web"})
        d = p.authorize(make_context(tool_name="read_file"))
        assert d.authorized is True

    def test_allowlist_block(self):
        p = ToolListGuardrailProvider(allowed_tools={"read_file"})
        d = p.authorize(make_context(tool_name="shell_exec"))
        assert d.authorized is False

    def test_denylist_block(self):
        p = ToolListGuardrailProvider(denied_tools={"shell_exec"})
        d = p.authorize(make_context(tool_name="shell_exec"))
        assert d.authorized is False

    def test_denylist_allow(self):
        p = ToolListGuardrailProvider(denied_tools={"shell_exec"})
        d = p.authorize(make_context(tool_name="read_file"))
        assert d.authorized is True

    def test_allow_overrides_deny(self):
        p = ToolListGuardrailProvider(
            allowed_tools={"shell_exec"},
            denied_tools={"shell_exec"},
        )
        d = p.authorize(make_context(tool_name="shell_exec"))
        assert d.authorized is True  # explicit allow wins

    def test_content_addressed_id(self):
        p = ToolListGuardrailProvider(allowed_tools={"a", "b"})
        d1 = p.authorize(make_context(tool_name="a"))
        d2 = p.authorize(make_context(tool_name="a"))
        assert d1.decision_id == d2.decision_id  # same input → same id


# ===========================================================================
# TestCKGGuardrailProvider
# ===========================================================================

class TestCKG:
    def test_tool_name_in_pass(self):
        p = CKGGuardrailProvider().add_constraint(
            "tool_name_in", tools={"read_file"}
        )
        d = p.authorize(make_context(tool_name="read_file"))
        assert d.authorized is True

    def test_tool_name_in_block(self):
        p = CKGGuardrailProvider().add_constraint(
            "tool_name_in", tools={"read_file"}
        )
        d = p.authorize(make_context(tool_name="shell"))
        assert d.authorized is False
        assert d.claims["failed_predicate"] == "tool_name_in"

    def test_tool_name_not_in(self):
        p = CKGGuardrailProvider().add_constraint(
            "tool_name_not_in", tools={"shell_exec", "rm"}
        )
        assert p.authorize(make_context(tool_name="read_file")).authorized is True
        assert p.authorize(make_context(tool_name="shell_exec")).authorized is False

    def test_agent_role_in(self):
        p = CKGGuardrailProvider().add_constraint(
            "agent_role_in", roles={"admin"}
        )
        assert p.authorize(make_context(agent_role="admin")).authorized is True
        assert p.authorize(make_context(agent_role="intern")).authorized is False

    def test_param_matches(self):
        p = CKGGuardrailProvider().add_constraint(
            "param_matches", name="mode", value="read"
        )
        assert p.authorize(
            make_context(tool_input={"mode": "read"})
        ).authorized is True
        assert p.authorize(
            make_context(tool_input={"mode": "write"})
        ).authorized is False

    def test_has_param(self):
        p = CKGGuardrailProvider().add_constraint("has_param", name="api_key")
        assert p.authorize(
            make_context(tool_input={"api_key": "xxx"})
        ).authorized is True
        assert p.authorize(make_context(tool_input={})).authorized is False

    def test_no_param(self):
        p = CKGGuardrailProvider().add_constraint("no_param", name="password")
        assert p.authorize(make_context(tool_input={})).authorized is True
        assert p.authorize(
            make_context(tool_input={"password": "secret"})
        ).authorized is False

    def test_multiple_constraints_and(self):
        p = (
            CKGGuardrailProvider()
            .add_constraint("tool_name_in", tools={"db_query"})
            .add_constraint("agent_role_in", roles={"admin"})
            .add_constraint("no_param", name="drop_table")
        )
        # All pass
        assert p.authorize(
            make_context(tool_name="db_query", agent_role="admin")
        ).authorized is True
        # One fails
        assert p.authorize(
            make_context(tool_name="db_query", agent_role="intern")
        ).authorized is False

    def test_chaining(self):
        p = (
            CKGGuardrailProvider()
            .add_constraint("tool_name_in", tools={"a"})
            .add_constraint("has_param", name="x")
        )
        # Assert on behaviour, not on the private _constraints list.
        # Tool "a" with param "x" -> both constraints satisfied.
        ctx_ok = ToolCallHookContext(tool_name="a", tool_input={"x": 1})
        assert p.authorize(ctx_ok).authorized is True
        # Tool "b" fails the tool_name_in constraint.
        ctx_bad_tool = ToolCallHookContext(tool_name="b", tool_input={"x": 1})
        assert p.authorize(ctx_bad_tool).authorized is False
        # Tool "a" but missing param "x" fails has_param.
        ctx_missing = ToolCallHookContext(tool_name="a", tool_input={})
        assert p.authorize(ctx_missing).authorized is False

    def test_unknown_predicate_rejected(self):
        """A misspelled predicate must raise, not silently allow."""
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError):
            p.add_constraint("tool_name_ins", tools={"a"})

    def test_missing_tools_argument_rejected(self):
        """add_constraint('tool_name_in') without tools= must raise at registration."""
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError, match="requires argument 'tools'"):
            p.add_constraint("tool_name_in")

    def test_missing_roles_argument_rejected(self):
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError, match="requires argument 'roles'"):
            p.add_constraint("agent_role_in")

    def test_missing_name_argument_rejected(self):
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError, match="requires argument 'name'"):
            p.add_constraint("has_param")

    def test_missing_value_argument_rejected(self):
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError, match="requires argument 'value'"):
            p.add_constraint("param_matches", name="mode")

    def test_empty_tools_argument_rejected(self):
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError, match="must not be empty"):
            p.add_constraint("tool_name_in", tools=set())

    def test_empty_name_argument_rejected(self):
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError, match="must not be empty"):
            p.add_constraint("has_param", name="")

    def test_wrong_type_tools_argument_rejected(self):
        p = CKGGuardrailProvider()
        with pytest.raises(ValueError, match="must be one of"):
            p.add_constraint("tool_name_in", tools="read_file")  # string not set

    def test_identical_calls_produce_distinct_envelopes(self):
        """Two executions with the same decision_id must both be retained."""
        trail = AuditTrail()
        claims = {"tool": "echo"}
        did = compute_decision_id(claims)
        trail.record_decision(
            GuardrailDecisionV1(
                decision_id=did, authorized=True, claims=claims
            )
        )
        e1 = ActionEnvelopeV1(
            decision_id=did,
            tool_result_digest="aaa",
            executed_at=1.0,
            duration_ms=1.0,
        )
        e2 = ActionEnvelopeV1(
            decision_id=did,
            tool_result_digest="bbb",
            executed_at=2.0,
            duration_ms=2.0,
        )
        trail.record_envelope(e1)
        trail.record_envelope(e2)
        # Both envelopes preserved.
        assert trail.envelope_count == 2
        all_e = trail.get_envelopes(did)
        assert len(all_e) == 2
        assert e1 in all_e and e2 in all_e
        # Default getter returns most recent.
        assert trail.get_envelope(did) is e2
        # Indexed access.
        assert trail.get_envelope(did, 0) is e1


# ===========================================================================
# TestGuardrailProviderProtocol
# ===========================================================================

class TestGuardrailProviderProtocol:
    def test_custom_provider(self):
        class AlwaysYes(GuardrailProvider):
            def authorize(self, context):
                return GuardrailDecisionV1(
                    decision_id=compute_decision_id({"custom": True}),
                    authorized=True,
                    claims={"custom": True},
                )

        p = AlwaysYes()
        d = p.authorize(make_context())
        assert d.authorized is True
        assert d.verify_integrity()

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            GuardrailProvider()


# ===========================================================================
# TestAuditTrail
# ===========================================================================

class TestAuditTrail:
    def test_record_and_get_decision(self):
        trail = AuditTrail()
        claims = {"test": 1}
        d = GuardrailDecisionV1(
            decision_id=compute_decision_id(claims),
            authorized=True, claims=claims,
        )
        trail.record_decision(d)
        assert trail.get_decision(d.decision_id) is d
        assert trail.decision_count == 1

    def test_record_and_get_envelope(self):
        trail = AuditTrail()
        e = ActionEnvelopeV1(
            decision_id="abc",
            tool_result_digest="def",
            executed_at=time.time(),
            duration_ms=1.0,
        )
        trail.record_envelope(e)
        assert trail.get_envelope("abc") is e
        assert trail.envelope_count == 1

    def test_get_nonexistent(self):
        trail = AuditTrail()
        assert trail.get_decision("nope") is None
        assert trail.get_envelope("nope") is None

    def test_clear(self):
        trail = AuditTrail()
        claims = {"x": 1}
        trail.record_decision(GuardrailDecisionV1(
            decision_id=compute_decision_id(claims),
            authorized=True, claims=claims,
        ))
        trail.record_envelope(ActionEnvelopeV1(
            decision_id="abc", tool_result_digest="d",
            executed_at=0, duration_ms=0,
        ))
        trail.clear()
        assert trail.decision_count == 0
        assert trail.envelope_count == 0


# ===========================================================================
# TestGuardrailContext
# ===========================================================================

class TestGuardrailContext:
    def test_allow(self):
        ctx = GuardrailContext(provider=AllowAllGuardrailProvider())
        d = ctx.authorize(make_context())
        assert d.authorized is True
        assert ctx.trail.decision_count == 1

    def test_block(self):
        ctx = GuardrailContext(provider=DenyAllGuardrailProvider())
        d = ctx.authorize(make_context())
        assert d.authorized is False

    def test_decision_recorded_in_trail(self):
        ctx = GuardrailContext(provider=AllowAllGuardrailProvider())
        d = ctx.authorize(make_context())
        assert ctx.trail.get_decision(d.decision_id) is d

    def test_on_deny_callback(self):
        denied = []
        ctx = GuardrailContext(
            provider=DenyAllGuardrailProvider(),
            on_deny=lambda d: denied.append(d),
        )
        ctx.authorize(make_context())
        assert len(denied) == 1
        assert denied[0].authorized is False

    def test_after_tool_call_envelope(self):
        ctx = GuardrailContext(provider=AllowAllGuardrailProvider())
        d = ctx.authorize(make_context())
        start = time.time()
        time.sleep(0.01)
        env = ctx.after_tool_call(d, result={"ok": True}, start_time=start)
        assert env.decision_id == d.decision_id
        assert env.duration_ms > 0
        assert ctx.trail.envelope_count == 1
        assert env.tool_result_digest == ActionEnvelopeV1.digest_result({"ok": True})

    def test_integrity_chain(self):
        """Decision integrity + envelope linkage."""
        ctx = GuardrailContext(provider=AllowAllGuardrailProvider())
        d = ctx.authorize(make_context())
        assert d.verify_integrity()
        env = ctx.after_tool_call(d, "result", time.time())
        assert env.decision_id == d.decision_id


# ===========================================================================
# TestMakeGuardrailHook
# ===========================================================================

class TestMakeGuardrailHook:
    def test_callable(self):
        hook = make_guardrail_hook(AllowAllGuardrailProvider())
        assert callable(hook)

    def test_returns_none_on_allow(self):
        hook = make_guardrail_hook(AllowAllGuardrailProvider())
        result = hook(make_context(tool_name="read_file"))
        assert result is None  # None = allow

    def test_returns_false_on_deny(self):
        hook = make_guardrail_hook(DenyAllGuardrailProvider())
        result = hook(make_context(tool_name="anything"))
        assert result is False  # False = block

    def test_stash_context(self):
        hook = make_guardrail_hook(AllowAllGuardrailProvider())
        assert hasattr(hook, "_guardrail_context")
        assert isinstance(hook._guardrail_context, GuardrailContext)

    def test_accumulate_decisions(self):
        trail = AuditTrail()
        hook = make_guardrail_hook(
            ToolListGuardrailProvider(allowed_tools={"a", "b"}),
            trail=trail,
        )
        hook(make_context(tool_name="a"))
        hook(make_context(tool_name="b"))
        hook(make_context(tool_name="c"))  # blocked
        assert trail.decision_count == 3


# ===========================================================================
# TestDetectMissingGuardrail (AS-GUARDRAIL-MISS-001)
# ===========================================================================

class TestDetectMissingGuardrail:
    def test_no_hooks_flags_critical(self):
        class FakeAgent:
            role = "researcher"

        # No global hooks registered -> CRITICAL finding.
        with _patch_global_hooks([]):
            findings = detect_missing_guardrail(FakeAgent())
        assert len(findings) == 1
        assert findings[0]["severity"] == "CRITICAL"
        assert findings[0]["pattern"] == "AS-GUARDRAIL-MISS-001"

    def test_unrelated_hook_still_flags(self):
        """An ordinary hook without _guardrail_context must NOT suppress the finding."""
        class FakeAgent:
            role = "researcher"

        def ordinary_hook(ctx):
            return None

        with _patch_global_hooks([ordinary_hook]):
            findings = detect_missing_guardrail(FakeAgent())
        assert len(findings) == 1
        assert findings[0]["severity"] == "CRITICAL"

    def test_with_guardrail_hook_no_finding(self):
        """A hook carrying the _guardrail_context marker suppresses the finding."""
        class FakeAgent:
            role = "researcher"

        def guardrail_hook(ctx):
            return None
        # Marker attached by make_guardrail_hook in production.
        guardrail_hook._guardrail_context = object()

        with _patch_global_hooks([guardrail_hook]):
            findings = detect_missing_guardrail(FakeAgent())
        assert findings == []

    def test_crew_with_multiple_agents_no_guardrail(self):
        class Agent:
            def __init__(self, role):
                self.role = role

        class Crew:
            agents = [Agent("a"), Agent("b"), Agent("c")]

        with _patch_global_hooks([]):
            findings = detect_missing_guardrail(Crew())
        # When no global guardrail hook exists, every agent is flagged.
        assert len(findings) == 3
        assert all(f["severity"] == "CRITICAL" for f in findings)


# ---------------------------------------------------------------------------
# Helper to patch the global before-tool-call hook list used by
# detect_missing_guardrail without requiring a full crewAI installation.
# ---------------------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def _patch_global_hooks(hooks):
    """Inject ``hooks`` as the result of ``get_before_tool_call_hooks()``.

    The import inside ``detect_missing_guardrail`` is lazy, so we patch at
    the module level where crewAI would normally expose it.  When crewAI
    is not installed we patch the fallback by injecting a fake module.
    """
    import sys
    import types

    fake_module = types.ModuleType("crewai.hooks.tool_hooks")
    fake_module.get_before_tool_call_hooks = lambda: list(hooks)
    fake_pkg = types.ModuleType("crewai")
    fake_pkg.__path__ = []  # mark as package
    fake_hooks_pkg = types.ModuleType("crewai.hooks")
    fake_hooks_pkg.__path__ = []
    sys.modules.setdefault("crewai", fake_pkg)
    sys.modules.setdefault("crewai.hooks", fake_hooks_pkg)
    sys.modules["crewai.hooks.tool_hooks"] = fake_module
    try:
        yield
    finally:
        sys.modules.pop("crewai.hooks.tool_hooks", None)


# ===========================================================================
# TestGuardrailScenario — 3 real-world scenarios end-to-end
# ===========================================================================

class TestGuardrailScenario:
    def test_scenario_1_deny_shell_in_research_crew(self):
        """Research crew: no agent should execute shell commands."""
        provider = ToolListGuardrailProvider(denied_tools={"shell_exec", "rm"})
        ctx = GuardrailContext(provider=provider)

        # Safe call
        d1 = ctx.authorize(make_context(tool_name="read_file"))
        assert d1.authorized is True

        # Dangerous call
        d2 = ctx.authorize(make_context(tool_name="shell_exec"))
        assert d2.authorized is False
        assert d2.verify_integrity()

    def test_scenario_2_role_based_db_access(self):
        """Only admin agents can query database."""
        provider = (
            CKGGuardrailProvider()
            .add_constraint("tool_name_in", tools={"db_query"})
            .add_constraint("agent_role_in", roles={"admin", "data_engineer"})
        )
        ctx = GuardrailContext(provider=provider)

        # Admin can query
        d1 = ctx.authorize(
            make_context(tool_name="db_query", agent_role="admin")
        )
        assert d1.authorized is True

        # Intern cannot
        d2 = ctx.authorize(
            make_context(tool_name="db_query", agent_role="intern")
        )
        assert d2.authorized is False

    def test_scenario_3_full_audit_chain(self):
        """End-to-end: authorize → execute → envelope → verify."""
        provider = ToolListGuardrailProvider(allowed_tools={"search_web"})
        trail = AuditTrail()
        ctx = GuardrailContext(provider=provider, trail=trail)

        # Authorize
        hook_ctx = make_context(tool_name="search_web")
        d = ctx.authorize(hook_ctx)
        assert d.authorized is True

        # Simulate execution
        start = time.time()
        result = {"results": ["link1", "link2"]}
        env = ctx.after_tool_call(d, result, start)

        # Verify chain
        assert d.verify_integrity()
        assert env.decision_id == d.decision_id
        assert trail.decision_count == 1
        assert trail.envelope_count == 1
        assert env.tool_result_digest == ActionEnvelopeV1.digest_result(result)
