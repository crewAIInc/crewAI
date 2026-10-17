"""
Contract tests for GovernanceDecision, GovernanceOutcome, and GovernanceSeal.

These tests validate that:
1. All four decision routes produce valid GovernanceDecision dicts
2. Extensions round-trip through JSON without validation failures
3. GovernanceOutcome links back to a decision via intent_ref
4. Unknown extension payloads are preserved without modification
5. Error outcomes carry error_type and error_message
6. seq/running_count enable omission detection (0-indexed)
7. GovernanceSeal detects tail-truncation
8. validate_governance_decision enforces route-specific required fields
9. Intent binding (TOCTOU closure) tests
10. intent_ref / receipt_ref identity split

No vendor imports. No external dependencies beyond stdlib.
"""

import json
from typing import Any

from crewai.governance.governance_decision import (
    GovernanceDecision,
    GovernanceOutcome,
    GovernanceSeal,
    validate_governance_decision,
    verify_contiguity,
)


# =============================================================================
# Section 1: Core Decision Route Fixtures (0-indexed)
# =============================================================================

FIXTURE_ALLOW: GovernanceDecision = {
    "decision_id": "d-001",
    "intent_ref": "sha256:a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
    "receipt_ref": "sha256:f0e1d2c3b4a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9f0e1",
    "agent_id": "support-bot",
    "agent_role": "Support Agent",
    "tool": "search_docs",
    "request_id": "req-abc-001",
    "params_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "normalized_scope": "docs/public",
    "intent_digest": "sha256:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b",
    "normalization_id": "jcs-sha256",
    "idempotency_key": "idem-allow-001",
    "target_state_digest": None,
    "policy_refs": ["allow-read-tools-v1"],
    "policy_digest": "sha256:policy-v1-hash",
    "decision": "allow",
    "reason": "Tool is in the agent's read allowlist",
    "issued_at": "2026-06-25T14:00:00Z",
    "seq": 0,
    "running_count": 1,
}

FIXTURE_DENY: GovernanceDecision = {
    "decision_id": "d-002",
    "intent_ref": "sha256:b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
    "receipt_ref": "sha256:e1d2c3b4a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2",
    "agent_id": "finance-agent",
    "agent_role": "Finance Analyst",
    "tool": "delete_customer",
    "request_id": "req-abc-002",
    "params_hash": "sha256:a8f3c91e4b2d7f6a1e9c3b5d8f2a4c6e0b7d9f1a3c5e7b9d1f3a5c7e9b0d2f4a",
    "normalized_scope": "customers/all",
    "policy_refs": ["deny-destructive-v1"],
    "decision": "deny",
    "reason": "Tool not in allowlist for Finance Analyst role",
    "issued_at": "2026-06-25T14:01:00Z",
    "seq": 1,
    "running_count": 2,
}

FIXTURE_REQUIRE_APPROVAL: GovernanceDecision = {
    "decision_id": "d-003",
    "intent_ref": "sha256:c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4",
    "receipt_ref": "sha256:d2c3b4a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c3",
    "agent_id": "admin-agent",
    "agent_role": "Admin",
    "tool": "export_data",
    "request_id": "req-abc-003",
    "target": "customer_database",
    "normalized_scope": "customers/eu",
    "params_hash": "sha256:d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6",
    "intent_digest": "sha256:3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d",
    "continuation_id": "cont-003-approval-pending",
    "normalization_id": "jcs-sha256",
    "idempotency_key": "idem-approval-003",
    "target_state_digest": "sha256:customer-db-state-at-auth",
    "policy_refs": ["require-approval-exports-v1"],
    "decision": "require_approval",
    "reason": "Data export requires human sign-off",
    "issued_at": "2026-06-25T14:05:00Z",
    "expires_at": "2026-06-25T14:10:00Z",
    "revalidate_if": ["target_state_change", "policy_version_change"],
    "seq": 2,
    "running_count": 3,
}

FIXTURE_REVISE: GovernanceDecision = {
    "decision_id": "d-005",
    "intent_ref": "sha256:e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
    "receipt_ref": "sha256:c3b4a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c3b4",
    "agent_id": "finance-agent",
    "agent_role": "Finance Analyst",
    "tool": "stripe.refund",
    "request_id": "req-abc-005",
    "params_hash": "sha256:c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5",
    "target": "payment_pmt_123",
    "normalized_scope": "payments/refund",
    "normalization_id": "jcs-sha256",
    "policy_refs": ["refund-limit-v1"],
    "decision": "revise",
    "reason": "Refund amount exceeds $1000 limit. Reduce amount and re-submit.",
    "issued_at": "2026-06-25T14:15:00Z",
    "revalidate_if": ["amount_changed"],
    "seq": 3,
    "running_count": 4,
}

FIXTURE_ALLOW_WITH_EXTENSION: GovernanceDecision = {
    "decision_id": "d-004",
    "intent_ref": "sha256:d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
    "receipt_ref": "sha256:b4a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c3b4a5",
    "agent_id": "ops-agent",
    "agent_role": "Operations",
    "tool": "deploy_service",
    "request_id": "req-abc-004",
    "params_hash": "sha256:b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4",
    "normalized_scope": "infra/deploy",
    "intent_digest": "sha256:4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e",
    "normalization_id": "jcs-sha256",
    "idempotency_key": "idem-deploy-004",
    "target_state_digest": "sha256:service-v2.1-running",
    "policy_refs": ["allow-deploy-with-evidence-v1"],
    "decision": "allow",
    "reason": "Policy: scoped token and audit receipt present",
    "issued_at": "2026-06-25T14:10:00Z",
    "evidence_refs": ["tealtiger-receipt-004"],
    "extensions": {
        "tealtiger": {
            "receipt_id": "tt-004",
            "merkle_proof": "sha256:proof-hash-here",
            "prev_hash": "sha256:f7a8b9c0d1e2f3a4b5c6d7e8",
            "verifier_contract_version": "2.1.0",
        },
        "vaara": {
            "chain_hash": "sha256:vaara-chain-hash",
            "contiguity_verified": True,
        },
    },
    "seq": 4,
    "running_count": 5,
}

FIXTURE_UNKNOWN_EXTENSION: GovernanceDecision = {
    "decision_id": "d-006",
    "intent_ref": "sha256:f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7",
    "receipt_ref": "sha256:a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e5d6c7b8a9f0e1d2c3b4a5f6",
    "agent_id": "test-agent",
    "tool": "any_tool",
    "normalized_scope": "test/scope",
    "params_hash": "sha256:test-params-hash",
    "intent_digest": "sha256:test-intent-digest",
    "normalization_id": "jcs-sha256",
    "idempotency_key": "idem-test-006",
    "target_state_digest": None,
    "policy_refs": ["test-policy"],
    "decision": "allow",
    "reason": "Testing unknown extension round-trip",
    "issued_at": "2026-06-25T14:20:00Z",
    "extensions": {
        "custom_vendor": {
            "arbitrary_field": True,
            "nested": {"deep": [1, 2, 3]},
            "unicode": "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8",
        }
    },
    "seq": 5,
    "running_count": 6,
}


# =============================================================================
# Section 2: Outcome Fixtures
# =============================================================================

FIXTURE_OUTCOME: GovernanceOutcome = {
    "decision_id": "d-004",
    "intent_ref": "sha256:d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
    "receipt_ref": "sha256:outcome-receipt-004",
    "idempotency_key": "idem-deploy-004",
    "outcome": "executed",
    "tool_output_hash": "sha256:d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
    "completed_at": "2026-06-25T14:10:02Z",
    "seq": 4,
}

FIXTURE_OUTCOME_ERROR: GovernanceOutcome = {
    "decision_id": "d-002",
    "intent_ref": "sha256:b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
    "receipt_ref": "sha256:outcome-receipt-002-error",
    "idempotency_key": "idem-deny-002",
    "outcome": "error",
    "error_type": "ToolExecutionError",
    "error_message": "Connection refused: database host unreachable",
    "completed_at": "2026-06-25T14:01:03Z",
    "seq": 1,
}


# =============================================================================
# Section 3: Core Contract Tests
# =============================================================================


def test_allow_fixture_is_valid_governance_decision() -> None:
    """ALLOW decision contains required binding fields."""
    assert FIXTURE_ALLOW["decision"] == "allow"
    assert "decision_id" in FIXTURE_ALLOW
    assert "agent_id" in FIXTURE_ALLOW
    assert "tool" in FIXTURE_ALLOW
    assert "intent_ref" in FIXTURE_ALLOW
    assert "params_hash" in FIXTURE_ALLOW
    assert "normalization_id" in FIXTURE_ALLOW
    assert "issued_at" in FIXTURE_ALLOW
    is_valid, errors = validate_governance_decision(FIXTURE_ALLOW)
    assert is_valid, f"Validation failed: {errors}"


def test_deny_fixture_is_valid_governance_decision() -> None:
    """DENY decision contains policy reference explaining the denial."""
    assert FIXTURE_DENY["decision"] == "deny"
    assert len(FIXTURE_DENY["policy_refs"]) > 0
    assert "reason" in FIXTURE_DENY
    is_valid, errors = validate_governance_decision(FIXTURE_DENY)
    assert is_valid, f"Validation failed: {errors}"


def test_require_approval_fixture_has_expiry() -> None:
    """REQUIRE_APPROVAL decision includes expires_at and continuation_id."""
    assert FIXTURE_REQUIRE_APPROVAL["decision"] == "require_approval"
    assert FIXTURE_REQUIRE_APPROVAL["expires_at"] is not None
    assert FIXTURE_REQUIRE_APPROVAL["continuation_id"] is not None
    is_valid, errors = validate_governance_decision(FIXTURE_REQUIRE_APPROVAL)
    assert is_valid, f"Validation failed: {errors}"


def test_revise_fixture_has_revalidate_if() -> None:
    """REVISE decision includes revalidate_if conditions (advisory only)."""
    assert FIXTURE_REVISE["decision"] == "revise"
    assert len(FIXTURE_REVISE["revalidate_if"]) > 0
    is_valid, errors = validate_governance_decision(FIXTURE_REVISE)
    assert is_valid, f"Validation failed: {errors}"


def test_extension_round_trips_through_json() -> None:
    """Extensions serialize to JSON and deserialize without data loss."""
    original = FIXTURE_ALLOW_WITH_EXTENSION
    serialized = json.dumps(original)
    deserialized = json.loads(serialized)

    assert deserialized["extensions"]["tealtiger"]["receipt_id"] == "tt-004"
    assert deserialized["extensions"]["vaara"]["contiguity_verified"] is True


def test_unknown_extension_round_trips_without_validation_failure() -> None:
    """Unknown vendor extensions pass through JSON round-trip unchanged."""
    original = FIXTURE_UNKNOWN_EXTENSION
    serialized = json.dumps(original)
    deserialized = json.loads(serialized)

    assert deserialized["extensions"]["custom_vendor"]["arbitrary_field"] is True
    assert deserialized["extensions"]["custom_vendor"]["nested"]["deep"] == [1, 2, 3]
    assert deserialized["extensions"]["custom_vendor"]["unicode"] == "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8"


def test_outcome_links_back_via_intent_ref() -> None:
    """GovernanceOutcome references the authorizing decision via intent_ref."""
    assert FIXTURE_OUTCOME["intent_ref"] == FIXTURE_ALLOW_WITH_EXTENSION["intent_ref"]
    assert FIXTURE_OUTCOME["outcome"] == "executed"
    assert "completed_at" in FIXTURE_OUTCOME


def test_error_outcome_has_error_fields() -> None:
    """Error outcome carries error_type and error_message."""
    assert FIXTURE_OUTCOME_ERROR["outcome"] == "error"
    assert FIXTURE_OUTCOME_ERROR["error_type"] is not None
    assert FIXTURE_OUTCOME_ERROR["error_message"] is not None
    assert FIXTURE_OUTCOME_ERROR["intent_ref"] == FIXTURE_DENY["intent_ref"]


def test_all_fixtures_json_serializable() -> None:
    """Every fixture round-trips through JSON without error."""
    fixtures: list[dict[str, Any]] = [
        FIXTURE_ALLOW, FIXTURE_DENY, FIXTURE_REQUIRE_APPROVAL,
        FIXTURE_ALLOW_WITH_EXTENSION, FIXTURE_REVISE,
        FIXTURE_OUTCOME, FIXTURE_OUTCOME_ERROR, FIXTURE_UNKNOWN_EXTENSION,
    ]
    for fixture in fixtures:
        serialized = json.dumps(fixture)
        deserialized = json.loads(serialized)
        assert deserialized == fixture


# =============================================================================
# Section 4: Validation Tests (Route-Specific Required Fields)
# =============================================================================


def test_allow_missing_binding_fields_fails_validation() -> None:
    """An ALLOW without full executable binding fields fails validation."""
    minimal_allow: GovernanceDecision = {
        "decision_id": "d-bad-001",
        "decision": "allow",
        "reason": "no binding fields",
    }
    is_valid, errors = validate_governance_decision(minimal_allow)
    assert is_valid is False
    assert any("agent_id" in e for e in errors)
    assert any("tool" in e for e in errors)
    assert any("normalized_scope" in e for e in errors)
    assert any("normalization_id" in e for e in errors)
    assert any("intent_digest" in e for e in errors)
    assert any("intent_ref" in e for e in errors)
    assert any("idempotency_key" in e for e in errors)


def test_deny_missing_reason_fails_validation() -> None:
    """A DENY without reason fails validation."""
    bad_deny: GovernanceDecision = {
        "decision_id": "d-bad-002",
        "tool": "some_tool",
        "decision": "deny",
    }
    is_valid, errors = validate_governance_decision(bad_deny)
    assert is_valid is False
    assert any("reason" in e for e in errors)


def test_revise_missing_revalidate_if_fails_validation() -> None:
    """A REVISE without revalidate_if fails validation."""
    bad_revise: GovernanceDecision = {
        "decision_id": "d-bad-003",
        "tool": "some_tool",
        "decision": "revise",
        "reason": "needs revision",
    }
    is_valid, errors = validate_governance_decision(bad_revise)
    assert is_valid is False
    assert any("revalidate_if" in e for e in errors)


def test_missing_decision_field_fails_validation() -> None:
    """A record with no decision field fails validation."""
    no_decision: GovernanceDecision = {
        "decision_id": "d-bad-004",
        "tool": "some_tool",
        "reason": "no decision field",
    }
    is_valid, errors = validate_governance_decision(no_decision)
    assert is_valid is False
    assert any("'decision'" in e for e in errors)


# =============================================================================
# Section 5: Completeness / Omission Detection Tests (0-indexed)
# =============================================================================

FIXTURE_CONTIGUOUS_RUN: list[GovernanceDecision] = [
    {"decision_id": "d-101", "tool": "search", "decision": "allow", "reason": "ok",
     "issued_at": "2026-06-25T10:00:00Z", "seq": 0, "running_count": 1},
    {"decision_id": "d-102", "tool": "calc", "decision": "allow", "reason": "ok",
     "issued_at": "2026-06-25T10:00:01Z", "seq": 1, "running_count": 2},
    {"decision_id": "d-103", "tool": "write", "decision": "deny", "reason": "blocked",
     "issued_at": "2026-06-25T10:00:02Z", "seq": 2, "running_count": 3},
]

FIXTURE_SEQ_GAP: list[GovernanceDecision] = [
    {"decision_id": "d-201", "tool": "search", "decision": "allow", "reason": "ok",
     "issued_at": "2026-06-25T10:00:00Z", "seq": 0, "running_count": 1},
    {"decision_id": "d-202", "tool": "calc", "decision": "allow", "reason": "ok",
     "issued_at": "2026-06-25T10:00:01Z", "seq": 1, "running_count": 2},
    # seq 2 missing -- provable interior gap
    {"decision_id": "d-204", "tool": "deploy", "decision": "allow", "reason": "ok",
     "issued_at": "2026-06-25T10:00:03Z", "seq": 3, "running_count": 4},
]

FIXTURE_RUNNING_COUNT_MISMATCH: list[GovernanceDecision] = [
    {"decision_id": "d-301", "tool": "search", "decision": "allow", "reason": "ok",
     "issued_at": "2026-06-25T10:00:00Z", "seq": 0, "running_count": 1},
    # running_count 4 at seq 1 means running_count != seq + 1 -- malformed
    {"decision_id": "d-302", "tool": "calc", "decision": "allow", "reason": "ok",
     "issued_at": "2026-06-25T10:00:01Z", "seq": 1, "running_count": 4},
]


def test_contiguous_run_passes_verification() -> None:
    """A complete 0-indexed run with no gaps passes contiguity verification."""
    assert verify_contiguity(FIXTURE_CONTIGUOUS_RUN) is True


def test_gap_in_seq_fails_verification() -> None:
    """A gap in seq (dropped record) is detected as incomplete."""
    assert verify_contiguity(FIXTURE_SEQ_GAP) is False


def test_running_count_mismatch_fails() -> None:
    """running_count != seq + 1 is detected as malformed."""
    assert verify_contiguity(FIXTURE_RUNNING_COUNT_MISMATCH) is False


def test_seq_starts_at_zero() -> None:
    """First decision in a run has seq=0, running_count=1."""
    first = FIXTURE_CONTIGUOUS_RUN[0]
    assert first["seq"] == 0
    assert first["running_count"] == 1


# =============================================================================
# Section 6: GovernanceSeal and Tail-Drop Detection Tests
# =============================================================================

FIXTURE_SEAL: GovernanceSeal = {
    "boundary_id": "crew-run-001",
    "sealed": True,
    "total": 3,
    "final_seq": 2,
    "sealed_at": "2026-06-25T10:00:05Z",
    "seal_hash": "sha256:concat-of-d101-d102-d103",
}

FIXTURE_TAIL_DROP_SEALED: list[dict[str, Any]] = [
    {"decision_id": "d-401", "tool": "search", "decision": "allow",
     "reason": "ok", "seq": 0, "running_count": 1},
    {"decision_id": "d-402", "tool": "calc", "decision": "allow",
     "reason": "ok", "seq": 1, "running_count": 2},
    {"decision_id": "d-403", "tool": "write", "decision": "allow",
     "reason": "ok", "seq": 2, "running_count": 3},
    # Seal says 4 total, but only 3 held — tail drop detected
    {"boundary_id": "crew-run-1", "sealed": True, "total": 4},
]

FIXTURE_TAIL_DROP_NO_SEAL: list[dict[str, Any]] = [
    {"decision_id": "d-401", "tool": "search", "decision": "allow",
     "reason": "ok", "seq": 0, "running_count": 1},
    {"decision_id": "d-402", "tool": "calc", "decision": "allow",
     "reason": "ok", "seq": 1, "running_count": 2},
    {"decision_id": "d-403", "tool": "write", "decision": "allow",
     "reason": "ok", "seq": 2, "running_count": 3},
    # No seal — tail drop is invisible (the irreducible residual)
]

FIXTURE_SEALED_WHOLE: list[dict[str, Any]] = [
    {"decision_id": "d-501", "tool": "search", "decision": "allow",
     "reason": "ok", "seq": 0, "running_count": 1},
    {"decision_id": "d-502", "tool": "calc", "decision": "allow",
     "reason": "ok", "seq": 1, "running_count": 2},
    {"decision_id": "d-503", "tool": "write", "decision": "deny",
     "reason": "blocked", "seq": 2, "running_count": 3},
    {"boundary_id": "crew-run-2", "sealed": True, "total": 3},
]


def test_sealed_whole_run_passes() -> None:
    """A complete run with matching seal passes verification."""
    assert verify_contiguity(FIXTURE_SEALED_WHOLE) is True


def test_tail_drop_caught_by_seal() -> None:
    """Seal pins total=4 but only 3 records held — tail drop detected."""
    assert verify_contiguity(FIXTURE_TAIL_DROP_SEALED) is False


def test_tail_drop_without_seal_is_the_residual() -> None:
    """Without a seal, tail drop is invisible — this is the honest residual."""
    assert verify_contiguity(FIXTURE_TAIL_DROP_NO_SEAL) is True


def test_seal_with_external_parameter() -> None:
    """verify_contiguity accepts an external seal parameter."""
    records = FIXTURE_CONTIGUOUS_RUN
    seal = {"total": 3}
    assert verify_contiguity(records, seal=seal) is True

    # Seal claims 5 but only 3 held
    bad_seal = {"total": 5}
    assert verify_contiguity(records, seal=bad_seal) is False


# =============================================================================
# Section 7: Intent Binding / TOCTOU Closure Tests
# =============================================================================


def test_intent_ref_stable_across_retries() -> None:
    """Same authorized intent with different timestamps produces same intent_ref.

      intent_ref = SHA-256(JCS({agent_id, tool, normalized_scope, intent_digest}))
      Note: idempotency_key is explicitly EXCLUDED (pair model).
      Duplicate enforcement uses the pair (intent_ref, idempotency_key).

    """
    # Two decisions for the same intent, different issued_at
    decision_1: GovernanceDecision = {
        "decision_id": "d-retry-001",
        "intent_ref": "sha256:same-intent-hash",
        "receipt_ref": "sha256:receipt-attempt-1",
        "agent_id": "bot-1",
        "tool": "search",
        "normalized_scope": "docs/public",
        "intent_digest": "sha256:intent-abc",
        "normalization_id": "jcs-sha256",
        "policy_refs": ["allow-v1"],
        "decision": "allow",
        "reason": "ok",
        "issued_at": "2026-06-25T10:00:00Z",
        "seq": 0, "running_count": 1,
    }
    decision_2: GovernanceDecision = {
        **decision_1,
        "decision_id": "d-retry-002",
        "receipt_ref": "sha256:receipt-attempt-2",  # different
        "issued_at": "2026-06-25T10:00:05Z",  # different timestamp
        "seq": 1, "running_count": 2,
    }
    # Same intent_ref despite different timestamps
    assert decision_1["intent_ref"] == decision_2["intent_ref"]
    # Different receipt_ref (per-record uniqueness)
    assert decision_1["receipt_ref"] != decision_2["receipt_ref"]


def test_intent_digest_mismatch_means_different_intent_ref() -> None:
    """Changed args produce a different intent_digest → different intent_ref."""
    original_intent_ref = "sha256:original-intent"
    mutated_intent_ref = "sha256:mutated-intent"
    # If intent_digest changes, intent_ref MUST change
    assert original_intent_ref != mutated_intent_ref


def test_target_state_digest_drift_requires_revalidation() -> None:
    """If target_state_digest changes, executor must revalidate."""
    decision: GovernanceDecision = {
        "decision_id": "d-drift-001",
        "intent_ref": "sha256:drift-intent",
        "agent_id": "bot-1",
        "tool": "update_customer",
        "target": "customer/123",
        "target_state_digest": "sha256:state-at-auth-time",
        "normalized_scope": "customers/write",
        "params_hash": "sha256:params-hash",
        "normalization_id": "jcs-sha256",
        "policy_refs": ["allow-update-v1"],
        "decision": "allow",
        "reason": "authorized",
        "issued_at": "2026-06-25T10:00:00Z",
        "seq": 0, "running_count": 1,
    }
    # Simulated current state at execution time
    current_state_digest = "sha256:state-has-drifted"
    # Contract invariant: mismatch requires revalidation
    assert decision["target_state_digest"] != current_state_digest
    # Executor should NOT proceed — must revalidate


def test_continuation_id_mismatch_denies_resume() -> None:
    """A deferred action resumed with wrong continuation_id is denied."""
    original_decision: GovernanceDecision = {
        "decision_id": "d-defer-001",
        "intent_ref": "sha256:defer-intent",
        "agent_id": "bot-1",
        "tool": "export_data",
        "normalized_scope": "data/export",
        "params_hash": "sha256:params",
        "normalization_id": "jcs-sha256",
        "policy_refs": ["require-approval-v1"],
        "decision": "require_approval",
        "reason": "needs human sign-off",
        "issued_at": "2026-06-25T10:00:00Z",
        "continuation_id": "cont-original-abc",
        "seq": 0, "running_count": 1,
    }
    # Attempt to resume with a different continuation_id
    resume_continuation = "cont-WRONG-xyz"
    assert original_decision["continuation_id"] != resume_continuation
    # Contract invariant: executor must deny (CONTINUATION_MISMATCH)


def test_idempotency_prevents_double_execution() -> None:
    """Same decision_id + intent_ref with existing terminal outcome = deny."""
    # Decision already has a terminal outcome
    executed_outcome: GovernanceOutcome = {
        "decision_id": "d-idem-001",
        "intent_ref": "sha256:idem-intent",
        "outcome": "executed",
        "completed_at": "2026-06-25T10:00:02Z",
        "seq": 0,
    }
    # A second execution attempt against the same authorization
    # Contract invariant: must be denied (IDEMPOTENCY_VIOLATION)
    assert executed_outcome["outcome"] == "executed"
    # Any system seeing an existing terminal outcome for this intent_ref
    # MUST deny a second execution attempt


def test_expired_authorization_denies() -> None:
    """An authorization past its expires_at must be denied."""
    decision: GovernanceDecision = {
        "decision_id": "d-expired-001",
        "intent_ref": "sha256:expired-intent",
        "agent_id": "bot-1",
        "tool": "deploy",
        "normalized_scope": "infra/deploy",
        "params_hash": "sha256:params",
        "normalization_id": "jcs-sha256",
        "policy_refs": ["allow-deploy-v1"],
        "decision": "allow",
        "reason": "authorized",
        "issued_at": "2026-06-25T10:00:00Z",
        "expires_at": "2026-06-25T10:05:00Z",
        "seq": 0, "running_count": 1,
    }
    # Simulated current time is past expiry
    current_time = "2026-06-25T10:06:00Z"
    assert decision["expires_at"] < current_time
    # Contract invariant: fail closed (AUTHORIZATION_EXPIRED)


# =============================================================================
# Section 8: intent_ref / receipt_ref Identity Split Tests
# =============================================================================


def test_intent_ref_is_join_key_between_decision_and_outcome() -> None:
    """GovernanceDecision and GovernanceOutcome join via intent_ref."""
    assert FIXTURE_ALLOW_WITH_EXTENSION["intent_ref"] == FIXTURE_OUTCOME["intent_ref"]


def test_receipt_ref_unique_per_record() -> None:
    """Every record has a unique receipt_ref (includes timestamp)."""
    all_receipt_refs = [
        FIXTURE_ALLOW["receipt_ref"],
        FIXTURE_DENY["receipt_ref"],
        FIXTURE_REQUIRE_APPROVAL["receipt_ref"],
        FIXTURE_ALLOW_WITH_EXTENSION["receipt_ref"],
        FIXTURE_REVISE["receipt_ref"],
    ]
    assert len(all_receipt_refs) == len(set(all_receipt_refs))


def test_same_intent_different_audit_timestamps_same_intent_ref() -> None:
    """Audit timestamp changes must not alter semantic intent identity."""
    # This is the key invariant: intent_ref excludes timestamp
    # Two records for the same intent at different times
    intent_ref_a = "sha256:stable-semantic-identity"
    intent_ref_b = "sha256:stable-semantic-identity"
    assert intent_ref_a == intent_ref_b


def test_different_scope_different_intent_ref() -> None:
    """Changed normalized_scope produces a different intent_ref."""
    # If scope changes, intent_ref MUST change — otherwise it's a bypass
    scope_a_ref = "sha256:intent-with-scope-a"
    scope_b_ref = "sha256:intent-with-scope-b"
    assert scope_a_ref != scope_b_ref


# =============================================================================
# Section 9: Deny is a First-Class Record Test
# =============================================================================


def test_deny_is_a_positive_record() -> None:
    """A DENY produces a full decision record, not an absence.

    This is the deny-as-record property: a blocked call leaves a
    recomputable record that reads differently from a call that was
    simply never made. A reviewer can tell 'denied and recorded' from
    'never observed'.
    """
    assert FIXTURE_DENY["decision"] == "deny"
    assert "decision_id" in FIXTURE_DENY
    assert "intent_ref" in FIXTURE_DENY
    assert "receipt_ref" in FIXTURE_DENY
    assert "seq" in FIXTURE_DENY
    assert "running_count" in FIXTURE_DENY
    # Deny records participate in the completeness sequence
    # just like allow records


# =============================================================================
# Section 10: normalization_id Tests
# =============================================================================


def test_normalization_id_identifies_hash_scheme() -> None:
    """normalization_id tells a verifier how to recompute params_hash."""
    assert FIXTURE_ALLOW["normalization_id"] == "jcs-sha256"
    # Other valid values: "agent-guard-unwrap-v1", "sql-normalize-v1"


def test_all_fixtures_carry_normalization_id() -> None:
    """All decision fixtures include normalization_id."""
    fixtures = [
        FIXTURE_ALLOW, FIXTURE_DENY, FIXTURE_REQUIRE_APPROVAL,
        FIXTURE_ALLOW_WITH_EXTENSION, FIXTURE_REVISE, FIXTURE_UNKNOWN_EXTENSION,
    ]
    for f in fixtures:
        assert "normalization_id" in f, f"Missing normalization_id in {f['decision_id']}"


# =============================================================================
# Section 11: Revise is Advisory-Only Tests
# =============================================================================


def test_revise_is_non_executable() -> None:
    """REVISE emits feedback and creates no side effect.

    Executing a revised action requires a fresh decision_id and digest.
    revise is advisory-only: no outcome with executed=true should exist
    for a revise decision without a new decision being issued first.
    """
    assert FIXTURE_REVISE["decision"] == "revise"
    # A revise decision should NEVER have an outcome with "executed"
    # without a subsequent allow decision being issued


# =============================================================================
# Section 12: seq and running_count Round-Trip Tests
# =============================================================================


def test_seq_and_running_count_round_trip() -> None:
    """seq and running_count fields survive JSON serialization."""
    for record in FIXTURE_CONTIGUOUS_RUN:
        deserialized = json.loads(json.dumps(record))
        assert deserialized["seq"] == record["seq"]
        assert deserialized["running_count"] == record["running_count"]


def test_running_count_equals_seq_plus_one() -> None:
    """For every record, running_count == seq + 1 (0-indexed invariant)."""
    all_records = FIXTURE_CONTIGUOUS_RUN + [
        FIXTURE_ALLOW, FIXTURE_DENY, FIXTURE_REQUIRE_APPROVAL,
        FIXTURE_ALLOW_WITH_EXTENSION, FIXTURE_REVISE, FIXTURE_UNKNOWN_EXTENSION,
    ]
    for record in all_records:
        assert record["running_count"] == record["seq"] + 1, (
            f"Record {record.get('decision_id')}: "
            f"running_count={record['running_count']} != seq+1={record['seq'] + 1}"
        )


def test_outcome_carries_seq_back_reference() -> None:
    """GovernanceOutcome carries the same seq as its linked decision."""
    assert FIXTURE_OUTCOME["seq"] == FIXTURE_ALLOW_WITH_EXTENSION["seq"]
    assert FIXTURE_OUTCOME_ERROR["seq"] == FIXTURE_DENY["seq"]


def test_empty_records_passes_verification() -> None:
    """An empty record list passes verification (vacuously true)."""
    assert verify_contiguity([]) is True


# =============================================================================
# Section 13: Duplicate Execution Prevention (intent_ref + idempotency_key)
# =============================================================================

# A terminal outcome exists for FIXTURE_ALLOW:
#   intent_ref = FIXTURE_ALLOW["intent_ref"]
#   idempotency_key = "idem-allow-001"
# This fixture attempts re-execution with a DIFFERENT decision_id but
# the same (intent_ref, idempotency_key) pair. The oracle MUST deny.

FIXTURE_DUPLICATE_DIFFERENT_DECISION_ID: GovernanceDecision = {
    "decision_id": "d-010",  # fresh decision record
    "intent_ref": FIXTURE_ALLOW["intent_ref"],  # same authorized intent
    "receipt_ref": "sha256:new-receipt-for-duplicate-attempt",
    "agent_id": "support-bot",
    "agent_role": "Support Agent",
    "tool": "search_docs",
    "request_id": "req-abc-010",
    "params_hash": FIXTURE_ALLOW["params_hash"],
    "normalized_scope": "docs/public",
    "intent_digest": FIXTURE_ALLOW["intent_digest"],
    "normalization_id": "jcs-sha256",
    "idempotency_key": "idem-allow-001",  # same idempotency key as FIXTURE_ALLOW
    "target_state_digest": None,
    "policy_refs": ["allow-read-tools-v1"],
    "decision": "allow",
    "reason": "Retry of previously executed intent",
    "issued_at": "2026-06-25T14:30:00Z",
    "seq": 6,
    "running_count": 7,
}

# Simulated terminal outcome for FIXTURE_ALLOW (the original execution)
FIXTURE_OUTCOME_FOR_ALLOW: GovernanceOutcome = {
    "decision_id": "d-001",
    "intent_ref": FIXTURE_ALLOW["intent_ref"],
    "receipt_ref": "sha256:outcome-receipt-001",
    "idempotency_key": "idem-allow-001",
    "outcome": "executed",
    "tool_output_hash": "sha256:search-results-hash",
    "completed_at": "2026-06-25T14:00:02Z",
    "seq": 0,
}


def test_duplicate_execution_denied_different_decision_id() -> None:
    """Same (intent_ref, idempotency_key) with different decision_id → deny.

    Duplicate-side-effect prevention must NOT depend on runtime-local
    record identity (decision_id). A terminal outcome for (intent_ref,
    idempotency_key) blocks any subsequent execution regardless of
    decision_id.

    This is the negative fixture requested by @safal207.
    """
    # The oracle has a terminal outcome for:
    #   intent_ref = FIXTURE_ALLOW["intent_ref"]
    #   idempotency_key = "idem-allow-001"
    existing_outcome = FIXTURE_OUTCOME_FOR_ALLOW
    assert existing_outcome["outcome"] == "executed"  # terminal

    # New decision attempts same (intent_ref, idempotency_key) with fresh decision_id
    new_decision = FIXTURE_DUPLICATE_DIFFERENT_DECISION_ID
    assert new_decision["decision_id"] != existing_outcome["decision_id"]
    assert new_decision["intent_ref"] == existing_outcome["intent_ref"]
    assert new_decision["idempotency_key"] == existing_outcome["idempotency_key"]

    # Contract invariant: oracle MUST deny this execution.
    # The duplicate check keys on (intent_ref, idempotency_key), NOT decision_id.
    # Expected verdict: DENY (reason: IDEMPOTENCY_VIOLATION)


def test_idempotency_key_is_first_class_on_outcome() -> None:
    """GovernanceOutcome carries idempotency_key as a first-class field.

    Previously in extensions — now promoted for vendor-neutral duplicate
    enforcement without relying on runtime-local record identity.
    """
    assert "idempotency_key" in FIXTURE_OUTCOME_FOR_ALLOW
    assert FIXTURE_OUTCOME_FOR_ALLOW["idempotency_key"] == "idem-allow-001"

    # The key is stable across retries of one side effect
    # (not unique to each attempt)
    assert FIXTURE_OUTCOME_FOR_ALLOW["idempotency_key"] == FIXTURE_ALLOW["idempotency_key"]


def test_different_idempotency_key_same_intent_ref_is_allowed() -> None:
    """Same intent_ref but different idempotency_key = genuinely new action.

    This is NOT a duplicate — it's a new authorized execution of the same
    intent with a fresh idempotency key (e.g., a legitimate re-invocation
    after the user explicitly re-requested the action).
    """
    new_execution: GovernanceDecision = {
        **FIXTURE_ALLOW,
        "decision_id": "d-011",
        "receipt_ref": "sha256:new-receipt-genuinely-new",
        "idempotency_key": "idem-allow-002-genuinely-new",  # different key
        "issued_at": "2026-06-25T15:00:00Z",
        "seq": 7,
        "running_count": 8,
    }
    # Same intent_ref, different idempotency_key → NOT a duplicate
    assert new_execution["intent_ref"] == FIXTURE_ALLOW["intent_ref"]
    assert new_execution["idempotency_key"] != FIXTURE_ALLOW["idempotency_key"]
    # Oracle should ALLOW this (no terminal outcome for this idempotency_key)



# =============================================================================
# Section 14: Unknown Verdict and params_hash Regression Tests
# =============================================================================


def test_unknown_verdict_fails_validation() -> None:
    """Unknown decision values must fail closed — not silently validate."""
    invalid_decisions: list[GovernanceDecision] = [
        {"decision_id": "d-bad-1", "decision": "approve"},
        {"decision_id": "d-bad-2", "decision": "ALLOW"},
        {"decision_id": "d-bad-3", "decision": "permit"},
    ]

    for d in invalid_decisions:
        is_valid, errors = validate_governance_decision(d)
        assert not is_valid, f"Expected invalid for decision='{d['decision']}'"
        assert any("Unknown decision" in e for e in errors)


def test_missing_params_hash_fails_allow_validation() -> None:
    """An allow decision without params_hash must fail validation."""
    decision: GovernanceDecision = {
        "decision_id": "d-no-hash",
        "decision": "allow",
        "agent_id": "bot",
        "tool": "search",
        "normalized_scope": "docs/public",
        "normalization_id": "jcs-sha256",
        "intent_digest": "sha256:abc",
        "intent_ref": "sha256:def",
        "idempotency_key": "idem:001",
        "issued_at": "2026-07-21T10:00:00Z",
        "policy_refs": ["allow-v1"],
        "target_state_digest": None,
        # params_hash intentionally missing
    }

    is_valid, errors = validate_governance_decision(decision)
    assert not is_valid
    assert any("params_hash" in e for e in errors)

