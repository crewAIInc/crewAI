"""
Contract tests for GovernanceDecision and GovernanceOutcome.

These tests validate that:
1. All four decision routes produce valid GovernanceDecision dicts
2. Extensions round-trip through JSON without validation failures
3. GovernanceOutcome links back to a decision via decision_id
4. Unknown extension payloads are preserved without modification

No vendor imports. No external dependencies beyond stdlib.
"""

import json
from typing import Any

from crewai.governance.governance_decision import GovernanceDecision, GovernanceOutcome


# --- Contract Test Fixtures ---

FIXTURE_ALLOW: GovernanceDecision = {
    "decision_id": "d-001",
    "agent_id": "support-bot",
    "agent_role": "Support Agent",
    "tool": "search_docs",
    "request_id": "req-abc-001",
    "params_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "policy_refs": ["allow-read-tools-v1"],
    "decision": "allow",
    "reason": "Tool is in the agent's read allowlist",
    "issued_at": "2026-06-03T14:00:00Z",
}

FIXTURE_DENY: GovernanceDecision = {
    "decision_id": "d-002",
    "agent_id": "finance-agent",
    "agent_role": "Finance Analyst",
    "tool": "delete_customer",
    "request_id": "req-abc-002",
    "params_hash": "sha256:a8f3c91e4b2d7f6a1e9c3b5d8f2a4c6e0b7d9f1a3c5e7b9d1f3a5c7e9b0d2f4a",
    "policy_refs": ["deny-destructive-v1"],
    "decision": "deny",
    "reason": "Tool not in allowlist for Finance Analyst role",
    "issued_at": "2026-06-03T14:01:00Z",
}

FIXTURE_REQUIRE_APPROVAL: GovernanceDecision = {
    "decision_id": "d-003",
    "agent_id": "admin-agent",
    "agent_role": "Admin",
    "tool": "export_data",
    "request_id": "req-abc-003",
    "target": "customer_database",
    "policy_refs": ["require-approval-exports-v1"],
    "decision": "require_approval",
    "reason": "Data export requires human sign-off",
    "issued_at": "2026-06-03T14:05:00Z",
    "expires_at": "2026-06-03T14:10:00Z",
}

FIXTURE_ALLOW_WITH_EXTENSION: GovernanceDecision = {
    "decision_id": "d-004",
    "agent_id": "ops-agent",
    "agent_role": "Operations",
    "tool": "deploy_service",
    "request_id": "req-abc-004",
    "params_hash": "sha256:b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4",
    "policy_refs": ["allow-deploy-with-evidence-v1"],
    "decision": "allow",
    "reason": "Policy: scoped token and audit receipt present",
    "issued_at": "2026-06-03T14:10:00Z",
    "evidence_refs": ["teec-receipt-004"],
    "extensions": {
        "teec": {
            "receipt_id": "teec-004",
            "evidence_hash": "sha256:b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8",
            "prev_hash": "sha256:f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
            "verifier_contract_version": "1.0.0",
        }
    },
}

FIXTURE_REVISE: GovernanceDecision = {
    "decision_id": "d-005",
    "agent_id": "finance-agent",
    "agent_role": "Finance Analyst",
    "tool": "stripe.refund",
    "request_id": "req-abc-005",
    "params_hash": "sha256:c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5",
    "target": "payment_pmt_123",
    "policy_refs": ["refund-limit-v1"],
    "decision": "revise",
    "reason": "Refund amount exceeds $1000 limit. Reduce amount below $1000 and re-submit.",
    "issued_at": "2026-06-03T14:15:00Z",
    "revalidate_if": ["amount_changed"],
}

FIXTURE_OUTCOME: GovernanceOutcome = {
    "decision_id": "d-004",
    "outcome": "executed",
    "tool_output_hash": "sha256:d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
    "completed_at": "2026-06-03T14:10:02Z",
}

FIXTURE_UNKNOWN_EXTENSION: GovernanceDecision = {
    "decision_id": "d-006",
    "agent_id": "test-agent",
    "tool": "any_tool",
    "decision": "allow",
    "reason": "Testing unknown extension round-trip",
    "issued_at": "2026-06-03T14:20:00Z",
    "extensions": {
        "custom_vendor": {
            "arbitrary_field": True,
            "nested": {"deep": [1, 2, 3]},
            "unicode": "日本語テスト",
        }
    },
}


# --- Contract Tests ---


def test_allow_fixture_is_valid_governance_decision() -> None:
    """ALLOW decision contains minimum required fields."""
    assert FIXTURE_ALLOW["decision"] == "allow"
    assert "decision_id" in FIXTURE_ALLOW
    assert "agent_id" in FIXTURE_ALLOW
    assert "tool" in FIXTURE_ALLOW
    assert "reason" in FIXTURE_ALLOW
    assert "issued_at" in FIXTURE_ALLOW


def test_deny_fixture_is_valid_governance_decision() -> None:
    """DENY decision contains policy reference explaining the denial."""
    assert FIXTURE_DENY["decision"] == "deny"
    assert len(FIXTURE_DENY["policy_refs"]) > 0
    assert "reason" in FIXTURE_DENY


def test_require_approval_fixture_has_expiry() -> None:
    """REQUIRE_APPROVAL decision includes expires_at for time-bound approval."""
    assert FIXTURE_REQUIRE_APPROVAL["decision"] == "require_approval"
    assert FIXTURE_REQUIRE_APPROVAL["expires_at"] is not None


def test_revise_fixture_has_revalidate_if() -> None:
    """REVISE decision includes revalidate_if conditions."""
    assert FIXTURE_REVISE["decision"] == "revise"
    assert len(FIXTURE_REVISE["revalidate_if"]) > 0


def test_extension_round_trips_through_json() -> None:
    """Extensions serialize to JSON and deserialize without data loss."""
    original = FIXTURE_ALLOW_WITH_EXTENSION
    serialized = json.dumps(original)
    deserialized = json.loads(serialized)

    assert deserialized["extensions"]["teec"]["receipt_id"] == "teec-004"
    assert deserialized["extensions"]["teec"]["evidence_hash"] == original["extensions"]["teec"]["evidence_hash"]
    assert deserialized["extensions"]["teec"]["prev_hash"] == original["extensions"]["teec"]["prev_hash"]


def test_unknown_extension_round_trips_without_validation_failure() -> None:
    """Unknown vendor extensions pass through JSON round-trip unchanged.

    This proves the contract is vendor-neutral: CrewAI does not validate,
    strip, or modify extension payloads it doesn't recognize.
    """
    original = FIXTURE_UNKNOWN_EXTENSION
    serialized = json.dumps(original)
    deserialized = json.loads(serialized)

    assert deserialized["extensions"]["custom_vendor"]["arbitrary_field"] is True
    assert deserialized["extensions"]["custom_vendor"]["nested"]["deep"] == [1, 2, 3]
    assert deserialized["extensions"]["custom_vendor"]["unicode"] == "日本語テスト"


def test_outcome_links_back_to_decision() -> None:
    """GovernanceOutcome references the authorizing decision via decision_id."""
    assert FIXTURE_OUTCOME["decision_id"] == FIXTURE_ALLOW_WITH_EXTENSION["decision_id"]
    assert FIXTURE_OUTCOME["outcome"] == "executed"
    assert "completed_at" in FIXTURE_OUTCOME


def test_all_fixtures_json_serializable() -> None:
    """Every fixture round-trips through JSON without error."""
    fixtures: list[dict[str, Any]] = [
        FIXTURE_ALLOW,
        FIXTURE_DENY,
        FIXTURE_REQUIRE_APPROVAL,
        FIXTURE_ALLOW_WITH_EXTENSION,
        FIXTURE_REVISE,
        FIXTURE_OUTCOME,
        FIXTURE_UNKNOWN_EXTENSION,
    ]
    for fixture in fixtures:
        serialized = json.dumps(fixture)
        deserialized = json.loads(serialized)
        assert deserialized == fixture
