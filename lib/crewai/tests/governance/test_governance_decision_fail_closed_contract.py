"""
Fail-closed contract fixtures for GovernanceDecision.

These tests are deliberately contract-level. They do not depend on a concrete
middleware hook implementation. Instead, they pin the expected behavior a
runtime/evaluator must preserve when binding an authorization record to an
executable candidate.

Invariant:
  authorization binds exact action + exact target state + exact continuation
  + non-duplicate outcome + exact idempotency key + same boundary (run)
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
    """Small test oracle for the fail-closed GovernanceDecision contract.

    Binding checks:
      1. decision must be "allow"
      2. agent_id, tool, target, normalized_scope must match exactly
      3. intent_ref, intent_digest, params_hash must match exactly
      4. continuation_id must match
      5. idempotency_key on candidate must match authorized key
      6. target_state_digest drift => revalidate
      7. duplicate (intent_ref, idempotency_key) terminal outcome => deny

    Duplicate enforcement keys on (intent_ref, idempotency_key) only.
    decision_id is NOT part of the duplicate predicate — a fresh decision_id
    with the same semantic side effect must still be denied.
    """
    existing_outcomes = existing_outcomes or []

    if decision.get("decision") != "allow":
        return "deny", "decision_not_allow"

    # Exact spatial binding: agent, tool, target, scope
    for field in ("agent_id", "tool", "target", "normalized_scope"):
        if decision.get(field) != candidate.get(field):
            return "deny", f"{field}_mismatch"

    # Exact intent binding: intent_ref, intent_digest, params_hash
    for field in ("intent_ref", "intent_digest", "params_hash"):
        if decision.get(field) and decision.get(field) != candidate.get(field):
            return "deny", "exact_intent_mismatch"

    # Continuation binding
    if decision.get("continuation_id") != candidate.get("continuation_id"):
        return "deny", "continuation_mismatch"

    # Idempotency key binding: the candidate MUST present the same key
    # that was authorized. A candidate with a different key cannot use
    # this decision's authorization.
    if decision.get("idempotency_key") != candidate.get("idempotency_key"):
        return "deny", "idempotency_key_mismatch"

    # Target state drift => revalidation required
    if decision.get("target_state_digest") != candidate.get("target_state_digest"):
        return "revalidate", "target_state_drift"

    # Duplicate enforcement: (intent_ref, idempotency_key) only.
    # decision_id is irrelevant — a fresh record for the same side effect is denied.
    for outcome in existing_outcomes:
        same_intent = outcome.get("intent_ref") == decision.get("intent_ref")
        same_idempotency = (
            outcome.get("idempotency_key") == decision.get("idempotency_key")
        )
        terminal = outcome.get("outcome") in {"executed", "blocked", "error", "timeout"}
        if terminal and same_intent and same_idempotency:
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
        "boundary_id": "crew-run-abc-001",
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


# --- Exact intent binding ---


def test_exact_intent_mismatch_denies() -> None:
    """Changed executable intent must deny, even if actor/tool/target match."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["intent_digest"] = "sha256:intent-digest-mutated"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "deny"
    assert reason == "exact_intent_mismatch"


def test_params_hash_mismatch_denies() -> None:
    """Changed params_hash must deny — exact recomputation boundary."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["params_hash"] = "sha256:params-mutated"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "deny"
    assert reason == "exact_intent_mismatch"


# --- Target state drift ---


def test_target_state_drift_revalidates() -> None:
    """Same action against changed target state requires revalidation."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["target_state_digest"] = "sha256:target-state-drifted"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "revalidate"
    assert reason == "target_state_drift"


# --- Continuation binding ---


def test_continuation_mismatch_denies() -> None:
    """Approved action cannot be replayed under another continuation."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["continuation_id"] = "cont:different-thread"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "deny"
    assert reason == "continuation_mismatch"


# --- Idempotency key binding ---


def test_candidate_idempotency_key_mismatch_denies() -> None:
    """Candidate presenting a different idempotency_key cannot use this authorization.

    The oracle must verify that the candidate's presented key matches
    the authorized key. A candidate that changes its key to a different value
    is not the same side effect — deny.
    """
    decision = base_allow_decision()
    candidate = matching_candidate()
    candidate["idempotency_key"] = "idem:send-summary:user@example.com:TAMPERED"

    verdict, reason = evaluate_contract_binding(decision, candidate)

    assert verdict == "deny"
    assert reason == "idempotency_key_mismatch"


# --- Duplicate outcome enforcement ---


def test_duplicate_outcome_idempotency_collision_denies() -> None:
    """A terminal outcome for the same (intent_ref, idempotency_key) blocks re-execution."""
    decision = base_allow_decision()
    candidate = matching_candidate()
    existing_outcome: GovernanceOutcome = {
        "decision_id": "d-fail-closed-001",
        "intent_ref": "sha256:intent-ref-approved",
        "receipt_ref": "sha256:outcome-receipt-001",
        "idempotency_key": "idem:send-summary:user@example.com:001",
        "outcome": "executed",
        "tool_output_hash": "sha256:tool-output-001",
        "completed_at": "2026-06-25T14:00:02Z",
        "boundary_id": "crew-run-abc-001",
        "seq": 0,
    }

    verdict, reason = evaluate_contract_binding(
        decision,
        candidate,
        existing_outcomes=[existing_outcome],
    )

    assert verdict == "deny"
    assert reason == "duplicate_outcome"


def test_duplicate_different_decision_id_denies() -> None:
    """Fresh decision_id with same (intent_ref, idempotency_key) is denied.

    This is the critical test: duplicate-side-effect prevention must NOT
    depend on runtime-local record identity (decision_id). A terminal outcome
    for (intent_ref, idempotency_key) blocks any subsequent execution
    regardless of decision_id.
    """
    decision = base_allow_decision()
    decision["decision_id"] = "d-replay-fresh-id"  # DIFFERENT decision_id
    candidate = matching_candidate()

    # Existing terminal outcome from the ORIGINAL decision_id
    existing_outcome: GovernanceOutcome = {
        "decision_id": "d-fail-closed-001",  # original decision_id
        "intent_ref": "sha256:intent-ref-approved",
        "receipt_ref": "sha256:outcome-receipt-001",
        "idempotency_key": "idem:send-summary:user@example.com:001",
        "outcome": "executed",
        "tool_output_hash": "sha256:tool-output-001",
        "completed_at": "2026-06-25T14:00:02Z",
        "boundary_id": "crew-run-abc-001",
        "seq": 0,
    }

    # The oracle must deny: same (intent_ref, idempotency_key) pair has
    # a terminal outcome, even though decision_id differs.
    verdict, reason = evaluate_contract_binding(
        decision,
        candidate,
        existing_outcomes=[existing_outcome],
    )

    assert verdict == "deny"
    assert reason == "duplicate_outcome"


# --- Genuinely new invocation ---


def test_different_idempotency_key_same_intent_is_allowed() -> None:
    """Same intent_ref but different idempotency_key = genuinely new action.

    Model A (pair model): intent_ref excludes idempotency_key.
    A different key represents a new invocation of the same semantic intent.
    """
    decision = base_allow_decision()
    decision["idempotency_key"] = "idem:send-summary:user@example.com:002"  # NEW key
    candidate = matching_candidate()
    candidate["idempotency_key"] = "idem:send-summary:user@example.com:002"

    # Existing outcome is for a DIFFERENT idempotency_key
    existing_outcome: GovernanceOutcome = {
        "decision_id": "d-fail-closed-001",
        "intent_ref": "sha256:intent-ref-approved",
        "receipt_ref": "sha256:outcome-receipt-001",
        "idempotency_key": "idem:send-summary:user@example.com:001",  # old key
        "outcome": "executed",
        "tool_output_hash": "sha256:tool-output-001",
        "completed_at": "2026-06-25T14:00:02Z",
        "boundary_id": "crew-run-abc-001",
        "seq": 0,
    }

    # Should ALLOW: different idempotency_key means genuinely new action
    verdict, reason = evaluate_contract_binding(
        decision,
        candidate,
        existing_outcomes=[existing_outcome],
    )

    assert verdict == "allow"
    assert reason == "contract_binding_ok"


# --- Cross-run splice rejection (boundary_id) ---


def test_cross_run_record_splice_detected_by_contiguity() -> None:
    """Records from different runs (different boundary_id) must fail contiguity.

    A missing seq=1 from run A cannot be replaced by a seq=1 record from run B.
    verify_contiguity() requires all records share the same boundary_id.
    """
    from crewai.governance.governance_decision import verify_contiguity

    # Records from two different runs spliced together
    records = [
        {
            "decision_id": "d-run-a-0",
            "boundary_id": "crew-run-A",
            "seq": 0,
            "running_count": 1,
        },
        {
            "decision_id": "d-run-b-1",
            "boundary_id": "crew-run-B",  # DIFFERENT run
            "seq": 1,
            "running_count": 2,
        },
        {
            "decision_id": "d-run-a-2",
            "boundary_id": "crew-run-A",
            "seq": 2,
            "running_count": 3,
        },
    ]

    # Must fail: mixed boundary_ids
    assert verify_contiguity(records) is False


def test_same_boundary_id_contiguity_passes() -> None:
    """Records from the same run with consistent boundary_id pass contiguity."""
    from crewai.governance.governance_decision import verify_contiguity

    records = [
        {
            "decision_id": "d-run-a-0",
            "boundary_id": "crew-run-A",
            "seq": 0,
            "running_count": 1,
        },
        {
            "decision_id": "d-run-a-1",
            "boundary_id": "crew-run-A",
            "seq": 1,
            "running_count": 2,
        },
        {
            "decision_id": "d-run-a-2",
            "boundary_id": "crew-run-A",
            "seq": 2,
            "running_count": 3,
        },
    ]

    assert verify_contiguity(records) is True


def test_seal_boundary_id_mismatch_fails() -> None:
    """Seal with different boundary_id than records must fail."""
    from crewai.governance.governance_decision import verify_contiguity

    records = [
        {
            "decision_id": "d-run-a-0",
            "boundary_id": "crew-run-A",
            "seq": 0,
            "running_count": 1,
        },
        {
            "decision_id": "d-run-a-1",
            "boundary_id": "crew-run-A",
            "seq": 1,
            "running_count": 2,
        },
    ]

    # Seal claims boundary_id from a DIFFERENT run
    seal = {"boundary_id": "crew-run-B", "sealed": True, "total": 2}

    assert verify_contiguity(records, seal=seal) is False



# --- Missing boundary_id fail-closed ---


def test_missing_boundary_id_fails_when_others_have_it() -> None:
    """A record without boundary_id in a set where others have it must fail.

    verify_contiguity() fails closed: if any record carries boundary_id,
    all records must carry it. A missing boundary_id is treated as a
    potential splice from an unidentified run.
    """
    from crewai.governance.governance_decision import verify_contiguity

    records = [
        {
            "decision_id": "d-run-a-0",
            "boundary_id": "crew-run-A",
            "seq": 0,
            "running_count": 1,
        },
        {
            "decision_id": "d-run-a-1",
            # boundary_id MISSING — must fail closed
            "seq": 1,
            "running_count": 2,
        },
        {
            "decision_id": "d-run-a-2",
            "boundary_id": "crew-run-A",
            "seq": 2,
            "running_count": 3,
        },
    ]

    assert verify_contiguity(records) is False


def test_seal_missing_boundary_id_fails_when_records_have_it() -> None:
    """A seal without boundary_id when records carry it must fail closed."""
    from crewai.governance.governance_decision import verify_contiguity

    records = [
        {
            "decision_id": "d-run-a-0",
            "boundary_id": "crew-run-A",
            "seq": 0,
            "running_count": 1,
        },
        {
            "decision_id": "d-run-a-1",
            "boundary_id": "crew-run-A",
            "seq": 1,
            "running_count": 2,
        },
    ]

    # Seal has NO boundary_id
    seal = {"sealed": True, "total": 2}

    assert verify_contiguity(records, seal=seal) is False


# --- Isolated params_hash regression ---


def test_only_params_hash_removed_fails_allow_validation() -> None:
    """Removing ONLY params_hash from a valid allow decision must fail.

    This isolates the params_hash invariant — the decision is otherwise
    fully valid with all other binding fields present.
    """
    from crewai.governance.governance_decision import validate_governance_decision

    # Start with a fully valid allow decision
    valid_allow = {
        "decision_id": "d-isolated-001",
        "decision": "allow",
        "agent_id": "bot-1",
        "tool": "search",
        "normalized_scope": "docs/public",
        "normalization_id": "jcs-sha256",
        "intent_digest": "sha256:abc123",
        "intent_ref": "sha256:def456",
        "idempotency_key": "idem:isolated-001",
        "params_hash": "sha256:params-valid",
        "issued_at": "2026-07-22T10:00:00Z",
        "policy_refs": ["allow-v1"],
        "target_state_digest": None,
    }

    # Verify it passes first
    is_valid, errors = validate_governance_decision(valid_allow)
    assert is_valid, f"Baseline should be valid: {errors}"

    # Now remove ONLY params_hash
    missing_hash = {k: v for k, v in valid_allow.items() if k != "params_hash"}

    is_valid, errors = validate_governance_decision(missing_hash)
    assert not is_valid
    assert any("params_hash" in e for e in errors)
    # Verify no OTHER fields caused failure (only params_hash)
    non_params_errors = [e for e in errors if "params_hash" not in e]
    assert len(non_params_errors) == 0, f"Unexpected errors: {non_params_errors}"
