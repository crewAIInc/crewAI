"""
GovernanceDecision — Vendor-neutral governance hook return type for CrewAI.

This module defines the serialized contract that crew-level governance hooks
(before_tool_call / after_tool_call) can optionally return. External governance
engines (TealTiger, Neura Relay, Vaara, etc.) implement this contract without
requiring CrewAI to depend on any vendor package.

The GovernanceDecision is the pre-execution authorization record.
The GovernanceOutcome is the post-execution result record, linked back
to the decision via decision_id.

Vendor-specific evidence (signed receipts, Merkle proofs, etc.) lives
under the `extensions` dict and is never validated by CrewAI core.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class GovernanceDecision(TypedDict, total=False):
    """Pre-execution authorization record returned by a governance hook.

    All fields are optional (total=False) to allow governance engines to
    populate only the fields they support. The minimum useful decision is
    {decision, reason}.

    Extensions are pass-through: CrewAI will serialize/deserialize them
    without validation, allowing any governance engine to attach its own
    evidence format (e.g., extensions["teec"], extensions["neura"]).
    """

    # Identity
    decision_id: str
    """Unique identifier for this decision. Used by GovernanceOutcome to link back."""

    agent_id: str
    """Identifier of the agent requesting the tool call."""

    agent_role: str
    """Role of the agent (e.g., 'Researcher', 'Admin')."""

    # Action context
    tool: str
    """Name of the tool being invoked."""

    request_id: str
    """Unique identifier for the specific tool call request."""

    params_hash: str
    """SHA-256 hash of the canonicalized (JCS) tool call parameters."""

    target: str
    """Target resource or entity the tool operates on, if known."""

    # Policy evaluation
    policy_refs: list[str]
    """List of policy rule identifiers that were evaluated."""

    decision: Literal["allow", "deny", "require_approval", "revise"]
    """The governance verdict for this tool call."""

    reason: str
    """Human-readable explanation of why this decision was made."""

    # Lifecycle
    issued_at: str
    """ISO 8601 timestamp of when this decision was issued."""

    expires_at: str | None
    """ISO 8601 timestamp after which this decision is invalid (fail-closed to deny)."""

    supersedes: str | None
    """decision_id of a prior decision that this one explicitly overrides."""

    revalidate_if: list[str]
    """Conditions that require re-evaluation before execution (e.g., 'policy_updated', 'budget_changed')."""

    # Evidence
    evidence_refs: list[str]
    """References to external evidence artifacts (URIs, hashes, receipt IDs)."""

    extensions: dict[str, Any]
    """Vendor-specific evidence. CrewAI passes this through without validation.

    Examples:
        extensions["teec"] = {"receipt_id": "...", "evidence_hash": "...", "prev_hash": "..."}
        extensions["neura"] = {"relay_id": "...", "action_card": "..."}
    """

    # Completeness evidence (omission detection)
    seq: int
    """Monotonic position of this decision within the crew run. No gaps allowed.

    A verifier holding N records can prove completeness: if seq values form
    a contiguous 1..N range, no records were dropped. A gap in seq is a
    provable omission without access to the issuer.
    """

    running_count: int
    """Total number of decisions emitted in this run so far (including this one).

    Must equal seq for the current record. If max(running_count) across held
    records exceeds the number of held records, at least one record was dropped.
    """


class GovernanceOutcome(TypedDict, total=False):
    """Post-execution result record linked to a GovernanceDecision.

    Emitted after the tool call completes (or fails). The decision_id
    links this outcome back to the authorization record that preceded it.
    """

    decision_id: str
    """Links back to the GovernanceDecision that authorized this execution."""

    outcome: Literal["executed", "blocked", "error", "timeout"]
    """What actually happened after the governance decision."""

    tool_output_hash: str | None
    """SHA-256 hash of the tool output (not the raw output itself)."""

    error_type: str | None
    """Error class name if outcome is 'error'."""

    error_message: str | None
    """Error message if outcome is 'error'."""

    completed_at: str
    """ISO 8601 timestamp of when execution completed."""

    extensions: dict[str, Any]
    """Vendor-specific post-execution evidence."""

    seq: int
    """Back-reference to the seq of the GovernanceDecision this outcome links to.

    Enables omission detection for outcomes: a missing outcome for a known
    decision seq is a provable gap.
    """
