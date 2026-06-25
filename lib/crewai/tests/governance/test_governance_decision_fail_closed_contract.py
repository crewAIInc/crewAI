"""
Fail-closed contract fixtures for GovernanceDecision.

These tests are deliberately contract-level. They do not depend on a concrete
middleware hook implementation. Instead, they pin the expected behavior a
runtime/evaluator must preserve when binding an authorization record to an
executable candidate.

Invariant:
    authorization binds exact action + exact target state + exact continuation
    + non-duplicate outcome
"""

from __future__ import annotations

from typing import Any, Literal

from crewai.governance.governance_decision import GovernanceDecision, GovernanceOutcome

BindingVerdict = Literal["allow", "deny", "revalidate"]


def evaluate_contract_binding(
    decision: GovernanceDecision,
    candidate: dict[str, Any],
    existing_outcomes: list[GovernanceOutcome] | None = None,
) -> tuple[BindingVerdict, str]:
    """Small test oracle for the fail-closed GovernanceDecision contract."""
    existing_outcomes = existing_outcomes or []

    if decision.get("decision") != "allow":
        return "deny", "decision_not_allow"

    for field in ("agent_id", "tool", "target", "normalized_scope"):
        if decision.get(field) != candidate.get(field):
            return "deny", f"{field}_mismatch"

    for field in ("intent_ref", "intent_digest", "params_hash"):
        if decision.get(field) and decision.get(field) != candidate.get(field):
            return "deny", "exact_intent_mismatch"

    if decision.get("continuation_id") != candidate.get("continuation_id"):
        return "deny", "continuation_mismatch"

    if decision.get("target_state_digest") != candidate.get("target_state_digest"):
        return "revalidate", "target_state_drift"

    for outcome in existing_outcomes:
        same_decision = outcome.get("decision_id") == decision.get("decision_id")
        same_intent = outcome.get("intent_ref") == decision.get("intent_ref")
        same_idempotency = (
            outcome.get("extensions", {}).get("idempotency_key")
            == decision.get("idempotency_key")
        )
        terminal = outcome.get("outcome") in {"executed", "blocked", "error", "timeout"}
        if terminal and same_decision and same_intent and same_idempotency:
            return "deny", "duplicate_outcome"

    return "allow", "contract_binding_ok"


def base_allow_decision() -> GovernanceDecision:
    return {
        "decision_id": "d-fail-closed-001",
        "intent_ref": "sha256:intent-ref-approved",
        "receipt_ref": "sha256:receipt-ref-approved",
        "agent_id": "support-bot",
        "tool": "send_email",
        "request_id": "req-fail-closed-001",
        "target": "email:user@example.com",
        "normalized_scope": "email/outbound/user-summary",
        "params_hash": "sha256:params-approved",
        "intent_digest": "sha256:intent-digest-approved",
        "target_state_digest": "sha256:target-state-at-authorization",
        "continuation_id": "cont:original-thread",
        "normalization_id": "jcs-sha256",
        "idempotency_key": "idem:send-summary:user@example.com:001",
        "policy_refs": ["allow-user-summary-email-v1"],
        "decision": "allow",
        "reason": "Authorized exact outbound summary email.",
        "issued_at": "2026-06-25T14:00:00Z",
        "seq": 0,
        "running_count": 1,
    }


def matching_candidate() -> dict[str, Any]:
    return {
        "agent_id": "support-bot",
        "tool": "send_email",
        "target": "email:user@example.com",
        "normalized_scope": "email/outbound/user-summary",
        "params_hash": "sha256:params-approved",
        "intent_ref": "sha256:intent-ref-approved",
        "intent_digest": "sha256:intent-digest-approved",
        "target_state_digest": "sha256:target-state-at-authorization",
        "continuation_id": "cont:original-thread",
        "idempotency_key": "idem:send-summary:user@example.com:001",
    }


def test_exact_intent_mismatch_denies() -> None:
    """Changed executable intent must deny, even if actor/tool/target match."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["intent_digest"] = "sha256:intent-digest-mutated"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "deny"
    assert reason == "exact_intent_mismatch"


def test_target_state_drift_revalidates() -> None:
    """Same action against changed target state requires revalidation."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["target_state_digest"] = "sha256:target-state-drifted"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "revalidate"
    assert reason == "target_state_drift"


def test_continuation_mismatch_denies() -> None:
    """Approved action cannot be replayed under another continuation."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["continuation_id"] = "cont:different-thread"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "deny"
    assert reason == "continuation_mismatch"


def test_duplicate_outcome_idempotency_collision_denies() -> None:
    """A terminal outcome for the same idempotency key blocks re-execution."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    existing_outcome: GovernanceOutcome = {
        "decision_id": "d-fail-closed-001",
        "intent_ref": "sha256:intent-ref-approved",
        "receipt_ref": "sha256:outcome-receipt-001",
        "outcome": "executed",
        "tool_output_hash": "sha256:tool-output-001",
        "completed_at": "2026-06-25T14:00:02Z",
        "seq": 0,
        "extensions": {
            "idempotency_key": "idem:send-summary:user@example.com:001",
        },
    }

    verdict, reason = evaluate_contract_binding(
        decision,
        candidate,
        existing_outcomes=[existing_outcome],
    )

    assert verdict == "deny"
    assert reason == "duplicate_outcome"
