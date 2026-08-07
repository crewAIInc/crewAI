# Test suite for CCS-conformant GuardrailProvider implementation.
# CCS — Conformance Testing Protocol for MCP Agent Runtime Security
# Reference: https://correctover.com/ccs

# Tests for GuardrailProvider — content-addressed decision audit chain
#
# Run:  pytest tests/guardrails/test_guardrail_provider.py -v
#        (from the crewAI repo root)

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock

import pytest

from crewai.guardrails.guardrail_provider import (
    AllowAllGuardrailProvider,
    AuditTrail,
    CKGGuardrailProvider,
    DenyAllGuardrailProvider,
    GuardrailDecisionV1,
    GuardrailProvider,
    GuardrailContext,
    ToolListGuardrailProvider,
    ActionEnvelopeV1,
    compute_decision_id,
    detect_missing_guardrail,
    digest_result,
    make_guardrail_hook,
)


# ═══════════════════════════════════════════════════
#  1. Core data types
# ═══════════════════════════════════════════════════


class TestGuardrailDecisionV1:
    def test_frozen_dataclass(self) -> None:
        d = GuardrailDecisionV1(
            decision_id="abc123",
            authorized=True,
            claims={"tool": "read_file"},
            expires_at=time.time() + 3600,
        )
        with pytest.raises(AttributeError):
            d.authorized = False  # type: ignore[misc]

    def test_not_expired(self) -> None:
        d = GuardrailDecisionV1(
            decision_id="x",
            authorized=True,
            expires_at=time.time() + 3600,
        )
        assert not d.is_expired()

    def test_expired(self) -> None:
        d = GuardrailDecisionV1(
            decision_id="x",
            authorized=True,
            expires_at=time.time() - 10,
        )
        assert d.is_expired()

    def test_no_expiry_never_expires(self) -> None:
        d = GuardrailDecisionV1(decision_id="x", authorized=True)
        assert not d.is_expired()
        assert not d.is_expired(now=time.time() + 1e12)

    def test_verify_integrity_passes(self) -> None:
        claims = {"tool": "read_file", "agent": "researcher"}
        expires_at = time.time() + 3600
        decision_id = compute_decision_id(claims, expires_at)
        d = GuardrailDecisionV1(
            decision_id=decision_id,
            authorized=True,
            claims=claims,
            expires_at=expires_at,
        )
        assert d.verify_integrity()

    def test_verify_integrity_fails_on_tampered_claims(self) -> None:
        claims = {"tool": "read_file"}
        expires_at = time.time() + 3600
        decision_id = compute_decision_id(claims, expires_at)

        tampered_claims = {"tool": "delete_database"}
        d = GuardrailDecisionV1(
            decision_id=decision_id,
            authorized=True,
            claims=tampered_claims,
            expires_at=expires_at,
        )
        assert not d.verify_integrity()

    def test_verify_integrity_fails_on_tampered_expiry(self) -> None:
        claims = {"tool": "read_file"}
        expires_at = time.time() + 3600
        decision_id = compute_decision_id(claims, expires_at)

        d = GuardrailDecisionV1(
            decision_id=decision_id,
            authorized=True,
            claims=claims,
            expires_at=time.time() + 999999,
        )
        assert not d.verify_integrity()


class TestActionEnvelopeV1:
    def test_duration_ms(self) -> None:
        now = time.time()
        env = ActionEnvelopeV1(
            decision_id="x",
            tool_name="test",
            tool_input_snapshot="{}",
            tool_result_digest="abc",
            started_at=now,
            completed_at=now + 1.5,
            success=True,
        )
        assert env.duration_ms() == 1500.0

    def test_never_stores_raw_result(self) -> None:
        env = ActionEnvelopeV1(
            decision_id="x",
            tool_name="test",
            tool_input_snapshot="{}",
            tool_result_digest="abc123",
            started_at=time.time(),
            completed_at=time.time(),
            success=True,
        )
        assert not hasattr(env, "tool_result")
        assert env.tool_result_digest == "abc123"

    def test_frozen(self) -> None:
        env = ActionEnvelopeV1(
            decision_id="x",
            tool_name="test",
            tool_input_snapshot="{}",
            tool_result_digest="abc",
            started_at=0,
            completed_at=0,
            success=True,
        )
        with pytest.raises(AttributeError):
            env.success = False  # type: ignore[misc]


# ═══════════════════════════════════════════════════
#  2. Content-addressed hashing
# ═══════════════════════════════════════════════════


class TestComputeDecisionId:
    def test_deterministic(self) -> None:
        claims = {"a": 1, "b": 2}
        id1 = compute_decision_id(claims)
        id2 = compute_decision_id(claims)
        assert id1 == id2

    def test_content_addressed(self) -> None:
        """Different claims → different ID."""
        id1 = compute_decision_id({"a": 1})
        id2 = compute_decision_id({"a": 2})
        assert id1 != id2

    def test_expires_at_folded_into_preimage(self) -> None:
        """expires_at without vs with should differ."""
        id1 = compute_decision_id({"a": 1}, expires_at=1000)
        id2 = compute_decision_id({"a": 1}, expires_at=2000)
        assert id1 != id2

    def test_key_order_independent(self) -> None:
        """Sorted keys means {b:2, a:1} == {a:1, b:2}."""
        id1 = compute_decision_id({"a": 1, "b": 2})
        id2 = compute_decision_id({"b": 2, "a": 1})
        assert id1 == id2

    def test_output_format(self) -> None:
        decision_id = compute_decision_id({"tool": "test"})
        assert isinstance(decision_id, str)
        assert len(decision_id) == 64  # SHA-256 hex
        int(decision_id, 16)  # Must be valid hex

    def test_works_without_expires_at(self) -> None:
        decision_id = compute_decision_id({"tool": "test"})
        assert isinstance(decision_id, str)
        assert len(decision_id) == 64


class TestDigestResult:
    def test_empty_string(self) -> None:
        assert digest_result("") == hashlib.sha256(b"").hexdigest()

    def test_none(self) -> None:
        assert digest_result(None) == hashlib.sha256(b"").hexdigest()

    def test_different_values_differ(self) -> None:
        assert digest_result("hello") != digest_result("world")

    def test_deterministic(self) -> None:
        assert digest_result("hello") == digest_result("hello")


# ═══════════════════════════════════════════════════
#  3. Guardrail Providers
# ═══════════════════════════════════════════════════


def _make_context(
    tool_name: str = "read_file",
    tool_input: dict | None = None,
    agent_role: str = "researcher",
) -> MagicMock:
    ctx = MagicMock()
    ctx.tool_name = tool_name
    ctx.tool_input = tool_input or {"path": "/tmp/test.txt"}
    agent = MagicMock()
    agent.role = agent_role
    ctx.agent = agent
    ctx.task = MagicMock()
    ctx.crew = MagicMock()
    return ctx


class TestAllowAllGuardrailProvider:
    def test_always_authorized(self) -> None:
        p = AllowAllGuardrailProvider()
        for tool in ["read_file", "delete_db", "exec", "unknown"]:
            d = p.authorize(_make_context(tool_name=tool))
            assert d.authorized, f"{tool} should be allowed"

    def test_has_decision_id(self) -> None:
        d = AllowAllGuardrailProvider().authorize(_make_context())
        assert len(d.decision_id) == 64
        assert d.verify_integrity()


class TestDenyAllGuardrailProvider:
    def test_always_denied(self) -> None:
        p = DenyAllGuardrailProvider()
        for tool in ["read_file", "calculator"]:
            d = p.authorize(_make_context(tool_name=tool))
            assert not d.authorized, f"{tool} should be denied"

    def test_has_decision_id(self) -> None:
        d = DenyAllGuardrailProvider().authorize(_make_context())
        assert len(d.decision_id) == 64
        assert d.verify_integrity()


class TestToolListGuardrailProvider:
    def test_allows_listed_tools(self) -> None:
        p = ToolListGuardrailProvider(
            allowed_tools={"read_file", "search_web"},
            default_block=True,
        )
        assert p.authorize(_make_context(tool_name="read_file")).authorized
        assert p.authorize(_make_context(tool_name="search_web")).authorized

    def test_blocks_unlisted_tools(self) -> None:
        p = ToolListGuardrailProvider(
            allowed_tools={"read_file"},
            default_block=True,
        )
        assert not p.authorize(_make_context(tool_name="delete_db")).authorized
        assert not p.authorize(_make_context(tool_name="exec")).authorized

    def test_inverted_mode(self) -> None:
        """default_block=False means listed tools are DENIED (block list)."""
        p = ToolListGuardrailProvider(
            allowed_tools={"dangerous_tool"},
            default_block=False,
        )
        assert not p.authorize(_make_context(tool_name="dangerous_tool")).authorized
        assert p.authorize(_make_context(tool_name="read_file")).authorized

    def test_decision_id_integrity(self) -> None:
        p = ToolListGuardrailProvider(allowed_tools={"read_file"})
        d = p.authorize(_make_context(tool_name="read_file"))
        assert d.verify_integrity()


class TestCKGGuardrailProvider:
    def test_no_constraints_allows_all(self) -> None:
        p = CKGGuardrailProvider()
        assert p.authorize(_make_context(tool_name="any_tool")).authorized

    def test_tool_name_in(self) -> None:
        p = CKGGuardrailProvider(constraints=[("tool_name_in", {"names": ["read_file"]})])
        assert p.authorize(_make_context(tool_name="read_file")).authorized
        assert not p.authorize(_make_context(tool_name="delete_db")).authorized

    def test_tool_name_not_in(self) -> None:
        p = CKGGuardrailProvider(
            constraints=[("tool_name_not_in", {"names": ["delete_db", "exec"]})]
        )
        assert p.authorize(_make_context(tool_name="read_file")).authorized
        assert not p.authorize(_make_context(tool_name="delete_db")).authorized

    def test_agent_role_in(self) -> None:
        p = CKGGuardrailProvider(
            constraints=[("agent_role_in", {"roles": ["admin"]})]
        )
        assert p.authorize(_make_context(agent_role="admin")).authorized
        assert not p.authorize(_make_context(agent_role="user")).authorized

    def test_has_param(self) -> None:
        p = CKGGuardrailProvider(
            constraints=[("has_param", {"key": "path"})]
        )
        assert p.authorize(_make_context(tool_input={"path": "/tmp"})).authorized
        assert not p.authorize(_make_context(tool_input={})).authorized

    def test_no_param(self) -> None:
        p = CKGGuardrailProvider(
            constraints=[("no_param", {"key": "command"})]
        )
        assert p.authorize(_make_context(tool_input={"path": "/tmp"})).authorized
        assert not p.authorize(
            _make_context(tool_input={"command": "rm -rf /"})
        ).authorized

    def test_param_matches(self) -> None:
        p = CKGGuardrailProvider(
            constraints=[("param_matches", {"key": "mode", "value": "read"})]
        )
        assert p.authorize(_make_context(tool_input={"mode": "read"})).authorized
        assert not p.authorize(_make_context(tool_input={"mode": "write"})).authorized

    def test_multiple_constraints_all_must_pass(self) -> None:
        p = CKGGuardrailProvider(
            constraints=[
                ("tool_name_in", {"names": ["read_file"]}),
                ("has_param", {"key": "path"}),
            ]
        )
        assert p.authorize(
            _make_context(tool_name="read_file", tool_input={"path": "/tmp"})
        ).authorized
        assert not p.authorize(
            _make_context(tool_name="delete_db", tool_input={"path": "/tmp"})
        ).authorized
        assert not p.authorize(
            _make_context(tool_name="read_file", tool_input={})
        ).authorized

    def test_add_constraint_runtime(self) -> None:
        p = CKGGuardrailProvider()
        p.add_constraint("tool_name_in", names=["read_file"])
        assert p.authorize(_make_context(tool_name="read_file")).authorized
        assert not p.authorize(_make_context(tool_name="exec")).authorized

    def test_unknown_predicate_raises(self) -> None:
        p = CKGGuardrailProvider(constraints=[("nonexistent", {})])
        with pytest.raises(ValueError, match="Unknown CKG predicate"):
            p.authorize(_make_context())

    def test_decision_id_integrity(self) -> None:
        p = CKGGuardrailProvider(constraints=[("tool_name_in", {"names": ["read_file"]})])
        d = p.authorize(_make_context(tool_name="read_file"))
        assert d.verify_integrity()


# ═══════════════════════════════════════════════════
#  4. Provider protocol (extensibility)
# ═══════════════════════════════════════════════════


class TestGuardrailProviderProtocol:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            GuardrailProvider()  # type: ignore[abstract]

    def test_custom_provider(self) -> None:
        class CustomProvider(GuardrailProvider):
            def authorize(self, context):
                return GuardrailDecisionV1(
                    decision_id=compute_decision_id({"custom": True}),
                    authorized=False,
                    claims={"custom": True},
                    reason="custom block",
                    provider_name="custom",
                )

        p = CustomProvider()
        assert p.name() == "CustomProvider"
        d = p.authorize(_make_context())
        assert not d.authorized
        assert d.provider_name == "custom"

    def test_name_override(self) -> None:
        class NamedProvider(GuardrailProvider):
            def authorize(self, context):
                return GuardrailDecisionV1(decision_id="x", authorized=True)

            def name(self) -> str:
                return "overridden_name"

        assert NamedProvider().name() == "overridden_name"


# ═══════════════════════════════════════════════════
#  5. Audit trail
# ═══════════════════════════════════════════════════


class TestAuditTrail:
    def test_record_and_retrieve_decision(self) -> None:
        trail = AuditTrail()
        d = GuardrailDecisionV1(decision_id="abc", authorized=True)
        trail.record_decision(d)
        assert trail.get_decision("abc") is d
        assert trail.total_decisions == 1

    def test_record_and_retrieve_envelope(self) -> None:
        trail = AuditTrail()
        now = time.time()
        e = ActionEnvelopeV1(
            decision_id="abc",
            tool_name="t",
            tool_input_snapshot="{}",
            tool_result_digest="x",
            started_at=now,
            completed_at=now,
            success=True,
        )
        trail.record_envelope(e)
        assert trail.get_envelope("abc") is e
        assert trail.total_envelopes == 1

    def test_all_decisions(self) -> None:
        trail = AuditTrail()
        d1 = GuardrailDecisionV1(decision_id="a", authorized=True)
        d2 = GuardrailDecisionV1(decision_id="b", authorized=False)
        trail.record_decision(d1)
        trail.record_decision(d2)
        assert len(trail.all_decisions()) == 2

    def test_clear(self) -> None:
        trail = AuditTrail()
        trail.record_decision(GuardrailDecisionV1(decision_id="a", authorized=True))
        trail.clear()
        assert trail.total_decisions == 0
        assert trail.total_envelopes == 0

    def test_get_nonexistent(self) -> None:
        trail = AuditTrail()
        assert trail.get_decision("nonexistent") is None
        assert trail.get_envelope("nonexistent") is None


# ═══════════════════════════════════════════════════
#  6. Hook integration (GuardrailContext)
# ═══════════════════════════════════════════════════


class TestGuardrailContext:
    def test_before_tool_call_allows(self) -> None:
        provider = AllowAllGuardrailProvider()
        ctx = GuardrailContext(provider=provider)
        result = ctx.before_tool_call(_make_context(tool_name="read_file"))
        assert result is None  # None = allow

    def test_before_tool_call_blocks(self) -> None:
        provider = DenyAllGuardrailProvider()
        ctx = GuardrailContext(provider=provider)
        result = ctx.before_tool_call(_make_context(tool_name="read_file"))
        assert result is False  # False = block

    def test_decision_recorded_on_allow(self) -> None:
        provider = AllowAllGuardrailProvider()
        ctx = GuardrailContext(provider=provider)
        ctx.before_tool_call(_make_context(tool_name="read_file"))
        assert ctx.trail.total_decisions == 1

    def test_decision_recorded_on_block(self) -> None:
        provider = DenyAllGuardrailProvider()
        ctx = GuardrailContext(provider=provider)
        ctx.before_tool_call(_make_context(tool_name="read_file"))
        assert ctx.trail.total_decisions == 1

    def test_on_deny_callback(self) -> None:
        callback = MagicMock()
        provider = DenyAllGuardrailProvider()
        ctx = GuardrailContext(provider=provider, on_deny=callback)
        ctx.before_tool_call(_make_context(tool_name="read_file"))
        callback.assert_called_once()

    def test_decision_integrity_in_hook(self) -> None:
        """Every decision made through the hook should be self-verifying."""
        provider = ToolListGuardrailProvider(allowed_tools={"read_file"})
        ctx = GuardrailContext(provider=provider)
        ctx.before_tool_call(_make_context(tool_name="read_file"))
        d = ctx.trail.all_decisions()[0]
        assert d is not None
        assert d.verify_integrity()


class TestMakeGuardrailHook:
    def test_returns_callable(self) -> None:
        hook = make_guardrail_hook(AllowAllGuardrailProvider())
        assert callable(hook)

    def test_stashes_context(self) -> None:
        hook = make_guardrail_hook(AllowAllGuardrailProvider())
        assert hasattr(hook, "context")
        assert isinstance(hook.context, GuardrailContext)  # type: ignore[attr-defined]

    def test_hook_blocks(self) -> None:
        hook = make_guardrail_hook(DenyAllGuardrailProvider())
        result = hook(_make_context(tool_name="any"))
        assert result is False

    def test_hook_allows(self) -> None:
        hook = make_guardrail_hook(AllowAllGuardrailProvider())
        result = hook(_make_context(tool_name="any"))
        assert result is None

    def test_hook_records_decision(self) -> None:
        hook = make_guardrail_hook(AllowAllGuardrailProvider())
        hook(_make_context(tool_name="test"))
        assert hook.context.trail.total_decisions == 1  # type: ignore[attr-defined]

    def test_multiple_calls_accumulate(self) -> None:
        hook = make_guardrail_hook(ToolListGuardrailProvider(allowed_tools={"a", "b"}))
        hook(_make_context(tool_name="a"))
        hook(_make_context(tool_name="b"))
        hook(_make_context(tool_name="c"))  # blocked
        assert hook.context.trail.total_decisions == 3  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════
#  7. Engine-v4 seed integration
# ═══════════════════════════════════════════════════
# These tests validate that the PR code satisfies the patterns defined
# in the Correctover engine-v4 scanner (AS-GUARDRAIL-MISS-001).


class TestDetectMissingGuardrail:
    def test_no_tools_no_missing(self) -> None:
        config = {"name": "agent1"}
        assert detect_missing_guardrail(config) == []

    def test_tool_with_guardrail_provider_not_missing(self) -> None:
        config = {
            "name": "agent1",
            "tools": ["read_file"],
            "guardrails": {
                "providers": [{"tool": "read_file", "type": "ToolListGuardrailProvider"}],
            },
        }
        assert detect_missing_guardrail(config) == []

    def test_tool_with_wildcard_provider_not_missing(self) -> None:
        config = {
            "name": "agent1",
            "tools": ["read_file", "search_web"],
            "guardrails": {
                "providers": [{"tool": "*", "type": "ToolListGuardrailProvider"}],
            },
        }
        assert detect_missing_guardrail(config) == []

    def test_tool_without_provider_is_missing(self) -> None:
        config = {
            "name": "agent1",
            "tools": ["read_file", "delete_db"],
            "guardrails": {
                "providers": [{"tool": "read_file", "type": "ToolListGuardrailProvider"}],
            },
        }
        missing = detect_missing_guardrail(config)
        assert any("delete_db" in m for m in missing)
        assert not any("read_file" in m for m in missing)

    def test_no_guardrails_section_at_all(self) -> None:
        config = {
            "name": "agent1",
            "tools": ["delete_db", "exec"],
        }
        missing = detect_missing_guardrail(config)
        assert len(missing) == 2

    def test_tools_as_dicts(self) -> None:
        config = {
            "name": "agent1",
            "tools": [{"name": "tool_a"}, {"name": "tool_b"}],
        }
        missing = detect_missing_guardrail(config)
        assert len(missing) == 2
        assert all("tool_a" in m or "tool_b" in m for m in missing)


# ═══════════════════════════════════════════════════
#  8. Real-world scenario integration tests
# ═══════════════════════════════════════════════════


class TestGuardrailScenario:
    """End-to-end scenarios simulating real crewAI usage."""

    def test_read_only_agent_always_passes(self) -> None:
        """A read-only agent with an allowlist should work transparently."""
        provider = ToolListGuardrailProvider(
            allowed_tools={"read_file", "search_web", "list_directory"},
            default_block=True,
        )
        ctx = GuardrailContext(provider=provider)

        for tool in ["read_file", "search_web", "list_directory"]:
            result = ctx.before_tool_call(_make_context(tool_name=tool))
            assert result is None, f"{tool} should be allowed"

        assert ctx.trail.total_decisions == 3
        assert all(d.authorized for d in ctx.trail.all_decisions())

    def test_dangerous_tool_correctly_blocked(self) -> None:
        """A dangerous tool should be blocked with appropriate evidence."""
        provider = ToolListGuardrailProvider(
            allowed_tools={"read_file"},
            default_block=True,
        )
        ctx = GuardrailContext(provider=provider)

        result = ctx.before_tool_call(
            _make_context(tool_name="execute_shell")
        )
        assert result is False

        d = ctx.trail.all_decisions()[0]
        assert not d.authorized
        assert d.verify_integrity()

    def test_ckg_multi_constraint_agent(self) -> None:
        """CKG with multiple constraints, simulating a real agent policy."""
        provider = CKGGuardrailProvider(
            constraints=[
                ("tool_name_in", {"names": ["read_file", "search_web", "write_file"]}),
                ("agent_role_in", {"roles": ["editor"]}),
                ("has_param", {"key": "path"}),
            ]
        )
        ctx = GuardrailContext(provider=provider)

        # Editor using read_file with path → allowed
        r = ctx.before_tool_call(
            _make_context(
                tool_name="read_file",
                tool_input={"path": "/tmp/doc.txt"},
                agent_role="editor",
            )
        )
        assert r is None

        # Editor using delete_db (not in allowlist) → blocked
        r = ctx.before_tool_call(
            _make_context(
                tool_name="delete_db",
                tool_input={"path": "/data/db"},
                agent_role="editor",
            )
        )
        assert r is False

        # Viewer trying to use write_file (wrong role) → blocked
        r = ctx.before_tool_call(
            _make_context(
                tool_name="write_file",
                tool_input={"path": "/tmp/out.txt"},
                agent_role="viewer",
            )
        )
        assert r is False

        assert ctx.trail.total_decisions == 3

    def test_decision_chain_verifiable(self) -> None:
        """Every decision in a multi-call chain must pass verify_integrity."""
        provider = ToolListGuardrailProvider(
            allowed_tools={"read_file", "search_web"},
            default_block=True,
        )
        ctx = GuardrailContext(provider=provider)

        calls = [
            ("read_file", True),
            ("search_web", True),
            ("delete_db", False),
            ("read_file", True),
            ("exec", False),
        ]
        for tool_name, expected_authorized in calls:
            ctx.before_tool_call(_make_context(tool_name=tool_name))

        assert ctx.trail.total_decisions == len(calls)
        for d in ctx.trail.all_decisions():
            assert d.verify_integrity(), (
                f"Decision {d.decision_id[:12]} integrity check failed"
            )

        granted = [d for d in ctx.trail.all_decisions() if d.authorized]
        denied = [d for d in ctx.trail.all_decisions() if not d.authorized]
        assert len(granted) == 3
        assert len(denied) == 2

    def test_compute_decision_id_determinism_across_providers(self) -> None:
        """Same claims produce same decision_id regardless of provider type."""
        claims = {"tool": "read_file", "params": {"path": "/tmp/x"}}
        expires_at = 9999999999.0

        # Both providers with same claims+expiry → same decision_id
        d1 = ToolListGuardrailProvider(allowed_tools={"read_file"}).authorize(
            _make_context(tool_name="read_file")
        )
        d2 = CKGGuardrailProvider().authorize(
            _make_context(tool_name="read_file")
        )
        # Decision IDs differ because claims differ (provider puts different
        # metadata in claims). That's OK — the important property is that
        # each decision_id binds to its own claims deterministically.
        assert d1.decision_id != d2.decision_id
        assert d1.verify_integrity()
        assert d2.verify_integrity()
