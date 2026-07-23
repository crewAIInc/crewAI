"""
GovernanceDecision -- Vendor-neutral governance contract types for CrewAI.

This module defines the serialized contract shape for governance authorization
records. A future reducer integration will consume these types from
before_tool_call / after_tool_call hooks; this PR establishes the wire format
only and does NOT modify existing hook dispatch behavior.

External governance engines (TealTiger, Neura Relay, Vaara, agent-guard,
AlgoVoi, etc.) implement this contract without requiring CrewAI to depend on
any vendor package.

The GovernanceDecision is the pre-execution authorization record.
The GovernanceOutcome is the post-execution result record, linked back
to the decision via decision_id and intent_ref.
The GovernanceSeal is the terminal record that pins the run's final count
for tail-drop detection.

Vendor-specific evidence (signed receipts, Merkle proofs, etc.) lives
under the `extensions` dict and is never validated by CrewAI core.

Canonicalization: All hash fields (params_hash, intent_digest, intent_ref,
receipt_ref, decision_context_hash) MUST be computed over RFC 8785 (JCS)
canonicalized JSON. json.dumps(sort_keys=True) is NOT JCS and diverges on
Unicode and non-integer fields. Use a compliant JCS library.

Index base: 0-indexed. The first decision in a run is seq=0, running_count=1.
This matches the Vaara reference implementation (vaara.receipt/v1 SPEC.md 5.3).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class GovernanceDecision(TypedDict, total=False):
    """Pre-execution authorization record returned by a governance hook.

    All fields are optional (total=False) to allow governance engines to
    populate only the fields they support. However, route-specific validation
    (via validate_governance_decision()) enforces that executable decisions
    carry the binding fields needed for safe verification.

    Extensions are pass-through: CrewAI will serialize/deserialize them
    without validation, allowing any governance engine to attach its own
    evidence format (e.g., extensions["tealtiger"], extensions["vaara"],
    extensions["agent_guard"], extensions["algovoi"]).
    """

    # --- Identity ---

    decision_id: str
    """Unique identifier for this decision record (runtime-local UUID).
    Used by GovernanceOutcome to link back."""

    intent_ref: str
    """Stable semantic identity of the authorized intent.
    SHA-256(JCS({agent_id, tool, normalized_scope, intent_digest})).
    No timestamp, no idempotency_key — retries of the same authorized intent
    produce the same hash regardless of attempt count.
    This is the normative cross-runtime join key between GovernanceDecision
    and GovernanceOutcome. Duplicate enforcement keys on the PAIR
    (intent_ref, idempotency_key) — not on intent_ref alone."""

    receipt_ref: str
    """Per-record unique identity for audit enumeration.
    SHA-256(JCS({...intent_ref_fields, issued_at})).
    Includes timestamp — distinct records always have distinct receipt_ref.
    Used for record counting and de-duplication across retries."""

    # --- Agent context ---

    agent_id: str
    """Identifier of the agent requesting the tool call."""

    agent_role: str
    """Role of the agent (e.g., 'Researcher', 'Admin')."""

    # --- Action context ---

    tool: str
    """Name of the tool being invoked."""

    request_id: str
    """Unique identifier for the specific tool call request."""

    params_hash: str
    """SHA-256 hash of the RFC 8785 (JCS) canonicalized tool call parameters.
    This is the hash of the requested form. See also intent_digest for the
    normalized executable form."""

    target: str
    """Target resource or entity the tool operates on, if known."""

    normalized_scope: str
    """Explicit scope of the action (e.g., 'customers/eu', 'prod/deploy').
    Must not fall back to tool name alone — missing scope fails closed."""

    # --- Intent binding (TOCTOU closure) ---

    intent_digest: str
    """SHA-256 over the normalized executable action envelope:
    (agent_id, tool, params_hash, target_state_digest).
    The executor MUST recompute this immediately before the side effect.
    Mismatch = fail closed (reason: INTENT_BINDING_MISMATCH)."""

    target_state_digest: str | None
    """Hash of the target resource state at authorization time. If the target
    state changed between authorization and execution, revalidation is required
    (reason: TARGET_STATE_DRIFT)."""

    continuation_id: str | None
    """For DEFER/REQUIRE_APPROVAL decisions: a resumption token. The deferred
    action can only execute with this specific continuation_id, which binds to
    the original intent. Mismatch = deny (reason: CONTINUATION_MISMATCH)."""

    normalization_id: str
    """Identifies which normalization was applied before computing params_hash
    and intent_digest. Examples:
      - 'jcs-sha256' (structured tool args, RFC 8785 canonical)
      - 'agent-guard-unwrap-v1' (shell command unwrapping; annex + vectors:
        https://github.com/XuebinMa/agent-guard/tree/spike/agent-guard-unwrap-v1/spikes/agent-guard-unwrap-v1)
      - 'sql-normalize-v1' (SQL query normalization)
    A verifier uses this to know how to recompute the digest."""

    idempotency_key: str
    """Key stable across retries of one side effect — NOT a key unique to each
    attempt. A second execution with the same (intent_ref, idempotency_key)
    pair is denied (IDEMPOTENCY_VIOLATION), regardless of decision_id.

    Duplicate enforcement keys on (intent_ref, idempotency_key). The
    idempotency_key is NOT an input to intent_ref computation — changing
    the key produces a genuinely new invocation of the same semantic intent."""

    # --- Policy evaluation ---

    policy_refs: list[str]
    """List of policy rule identifiers that were evaluated."""

    retrieved_policy_refs: list[str]
    """Stable refs to policy or memory records consulted (for adaptive governance)."""

    policy_digest: str
    """Hash of the actual policy version evaluated."""

    decision: Literal["allow", "deny", "require_approval", "revise"]
    """The governance verdict for this tool call.

    Semantics:
      - allow: executable, binding fields required
      - deny: non-executable, recorded as first-class positive record
      - require_approval: blocked until approval produces a valid decision
      - revise: advisory feedback only; NO side effect; revised action requires
        a new normalized envelope, new digest, and new decision_id.
        Engines that don't implement revise simply never emit it.
    """

    reason: str
    """Human-readable explanation of why this decision was made."""

    # --- Lifecycle ---

    issued_at: str
    """ISO 8601 timestamp of when this decision was issued."""

    expires_at: str | None
    """ISO 8601 timestamp after which this decision is invalid (fail-closed to deny)."""

    supersedes: str | None
    """decision_id of a prior decision that this one explicitly overrides."""

    revalidate_if: list[str]
    """Conditions that require re-evaluation before execution.
    Examples: ['argument_change', 'target_state_change', 'budget_change',
    'policy_version_change', 'scope_expansion', 'agent_identity_rotation']"""

    # --- Context ---

    decision_context_hash: str
    """SHA-256 digest over JCS-canonicalized:
    {agent_id, tool, params_hash, intent_digest, seq, retrieved_policy_refs,
    policy_digest, credential_scope, credential_tier, expires_at, revalidate_if}.
    Enables drift detection: if any input changed, the hash changes."""

    credential_scope: str
    """Authority scope available to the agent (e.g., 'read-only', 'production-write')."""

    credential_tier: str
    """Credential tier level (e.g., 'service-account', 'human-delegated')."""

    # --- Evidence ---

    evidence_refs: list[str]
    """References to external evidence artifacts (URIs, hashes, receipt IDs)."""

    extensions: dict[str, Any]
    """Vendor-specific evidence. CrewAI passes this through without validation.

    Examples:
        extensions["tealtiger"] = {"receipt_id": "...", "merkle_proof": "..."}
        extensions["vaara"] = {"chain_hash": "...", "contiguity_report": "..."}
        extensions["agent_guard"] = {"decision_code": "DENIED_BY_RULE", "attestation": "..."}
        extensions["algovoi"] = {"keystone_ref": "...", "jcs_vectors": "..."}
        extensions["neura"] = {"relay_id": "...", "action_card": "..."}
    """

    # --- Completeness evidence (omission detection) ---

    boundary_id: str
    """Run/session identifier that scopes all seq-bearing records.
    Every GovernanceDecision, GovernanceOutcome, and GovernanceSeal in the
    same governed run MUST share the same boundary_id. A verifier MUST
    require equality between record boundary_id and seal boundary_id.
    This prevents cross-run record splicing: a record from run A cannot
    satisfy a gap in run B because boundary_id will mismatch.
    Kept out of intent_ref so semantic identity remains cross-run stable."""

    seq: int
    """0-indexed monotonic position of this decision within the crew run.
    First decision is seq=0. No gaps allowed.

    A verifier holding N records can detect internal gaps: if seq values do not
    form a contiguous 0..N-1 range, records were dropped. Note: seq + running_count
    detect INTERNAL gaps only. Tail-truncation detection requires a GovernanceSeal.
    """

    running_count: int
    """Total number of decisions emitted in this run so far (including this one).
    MUST equal seq + 1 for every record.

    If running_count != seq + 1, the record is malformed. If max(running_count)
    across held records exceeds the number of held records, at least one record
    was dropped.
    """


class GovernanceOutcome(TypedDict, total=False):
    """Post-execution result record linked to a GovernanceDecision.

    Emitted after the tool call completes (or fails). The intent_ref
    links this outcome back to the authorization record that preceded it.
    decision_id provides backward compatibility.
    """

    decision_id: str
    """Links back to the GovernanceDecision that authorized this execution."""

    intent_ref: str
    """Same intent_ref as the GovernanceDecision. This is the normative join key.
    A verifier recomputes intent_ref from the decision-side fields and confirms
    the outcome references the same authorized intent."""

    receipt_ref: str
    """Unique per-record identity for this outcome (includes timestamp)."""

    idempotency_key: str
    """Stable key identifying this specific side effect across retries.
    A key stable across retries of one side effect — NOT a key unique to
    each attempt. Duplicate detection keys on (intent_ref, idempotency_key):
    a terminal outcome for this pair blocks any subsequent execution
    regardless of decision_id.

    Promoted to first-class field (previously in extensions) to enable
    vendor-neutral duplicate enforcement without relying on runtime-local
    record identity."""

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

    # --- Completeness / run binding ---

    boundary_id: str
    """Same run/session identifier as the GovernanceDecision and GovernanceSeal.
    A verifier MUST require equality across all records in the same run.
    Prevents cross-run record splicing."""

    seq: int
    """Same seq value as the GovernanceDecision this outcome links to.
    Enables omission detection for outcomes: a missing outcome for a known
    decision seq is a provable gap."""


class GovernanceSeal(TypedDict, total=False):
    """Terminal record emitted at the end of a governed run/session.

    Pins the run's final decision count so that tail-truncation (records
    dropped from the end) is detectable — which per-record running_count
    alone cannot catch.

    Layering:
      1. seq → ordering (0-indexed)
      2. running_count == seq + 1 → per-record consistency
      3. hash chain (extensions) → tamper-evidence
      4. GovernanceSeal → tail-drop detection (total pins expected count)
      5. RFC 3161 external anchor (extensions, optional) → residual closure

    Honest residual: a suffix drop that ALSO suppresses the seal stays
    invisible from the held set alone. No field can close that. An external
    anchor is required for full residual closure.
    """

    boundary_id: str
    """Identifier for the run/session this seal covers (e.g., crew_run_id).
    MUST equal boundary_id on every seq-bearing record in the same run."""

    sealed: Literal[True]
    """Always True. Distinguishes seal records from decision records."""

    total: int
    """Total number of GovernanceDecision records emitted in this run.
    A verifier expects exactly this many seq-bearing records (0..total-1).
    If len(held_records) < total, at least (total - len) were dropped."""

    final_seq: int
    """The last seq value emitted before this seal. Should equal total - 1."""

    sealed_at: str
    """ISO 8601 timestamp of when the session was finalized."""

    seal_hash: str | None
    """Optional: SHA-256 digest of the concatenation of all decision_ids
    in sequence order. Provides tamper-evidence for the seal itself."""

    extensions: dict[str, Any]
    """Vendor-specific seal evidence (e.g., RFC 3161 timestamp token)."""


# --- Validation ---


_VALID_VERDICTS = {"allow", "deny", "require_approval", "revise"}


def validate_governance_decision(d: GovernanceDecision) -> tuple[bool, list[str]]:
    """Validate a GovernanceDecision has the required fields for its route.

    The TypedDict is total=False (wire format flexibility), but this validator
    enforces route-specific minimums so that executable decisions carry the
    full recomputation boundary needed for safe verification.

    This is a normative authorization validator — not just shape validation.
    An executable decision (allow / require_approval) MUST carry the complete
    binding fields that a future executor needs to recompute and verify the
    authorized intent before performing the side effect.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: list[str] = []

    decision = d.get("decision")
    if not decision:
        errors.append("'decision' field is required")
        return (False, errors)

    # Reject unknown verdicts — fail closed on unrecognized decision routes
    if decision not in _VALID_VERDICTS:
        errors.append(
            f"Unknown decision '{decision}' — must be one of: {sorted(_VALID_VERDICTS)}"
        )
        return (False, errors)

    # All routes need at minimum:
    if not d.get("decision_id"):
        errors.append(f"'{decision}' requires 'decision_id'")

    if decision == "allow":
        # Full executable binding: all fields needed to reconstruct the
        # authorized intent boundary at execution time.
        required = [
            "agent_id", "tool", "normalized_scope", "normalization_id",
            "intent_digest", "intent_ref", "idempotency_key",
            "params_hash", "issued_at",
        ]
        for field in required:
            if not d.get(field):
                errors.append(f"'allow' requires '{field}'")

        # policy_refs must have at least one entry
        if not d.get("policy_refs"):
            errors.append("'allow' requires at least one entry in 'policy_refs'")

        # target_state_digest key must be present (value may be None)
        if "target_state_digest" not in d:
            errors.append(
                "'allow' requires 'target_state_digest' key (value may be None)"
            )

    elif decision == "require_approval":
        # Same binding as allow + resume fields for safe continuation
        required = [
            "agent_id", "tool", "normalized_scope", "normalization_id",
            "intent_digest", "intent_ref", "idempotency_key",
            "params_hash", "issued_at",
            "continuation_id", "expires_at",
        ]
        for field in required:
            if not d.get(field):
                errors.append(f"'require_approval' requires '{field}'")

        # policy_refs must have at least one entry
        if not d.get("policy_refs"):
            errors.append(
                "'require_approval' requires at least one entry in 'policy_refs'"
            )

        # target_state_digest key must be present (value may be None)
        if "target_state_digest" not in d:
            errors.append(
                "'require_approval' requires 'target_state_digest' key (value may be None)"
            )

    elif decision == "deny":
        if not d.get("tool"):
            errors.append("'deny' requires 'tool'")
        if not d.get("reason"):
            errors.append("'deny' requires 'reason'")

    elif decision == "revise":
        if not d.get("tool"):
            errors.append("'revise' requires 'tool'")
        if not d.get("reason"):
            errors.append("'revise' requires 'reason'")
        if not d.get("revalidate_if"):
            errors.append("'revise' requires 'revalidate_if' conditions")

    return (len(errors) == 0, errors)


# --- Contiguity Verification ---


def verify_contiguity(
    records: list[dict[str, Any]],
    seal: dict[str, Any] | None = None,
) -> bool:
    """Verify that records form a complete, gap-free 0-indexed sequence.

    Checks:
      1. All records share the same boundary_id (no cross-run splicing)
      2. seq values form a contiguous 0..N-1 range (no gaps, no duplicates)
      3. running_count == seq + 1 for every record
      4. len(seq_records) == expected count

    When a seal is provided, additionally checks that:
      - len(seq_records) == seal["total"]
      - seal boundary_id matches record boundary_id

    This catches tail-truncation that per-record fields alone cannot detect.

    Returns True if complete. Returns False if any gap, duplicate, count
    mismatch, boundary_id mismatch, or seal violation exists.

    NOTE: This detects internal gaps and (with seal) tail drops. It CANNOT
    detect a suffix drop that also suppresses the seal — that requires an
    external anchor (RFC 3161 timestamp or equivalent).
    """
    from collections import Counter

    # Separate seal records from decision records
    seq_records = [r for r in records if not r.get("sealed")]
    sealed_records = [r for r in records if r.get("sealed")]

    # Determine expected count
    sealed_total = max(
        (int(r["total"]) for r in sealed_records), default=0
    )

    if seal is not None:
        sealed_total = max(sealed_total, int(seal.get("total", 0)))

    if not seq_records:
        return sealed_total == 0

    # Verify boundary_id consistency across all records
    boundary_ids = {r.get("boundary_id") for r in seq_records if r.get("boundary_id")}
    if seal and seal.get("boundary_id"):
        boundary_ids.add(seal["boundary_id"])
    for sr in sealed_records:
        if sr.get("boundary_id"):
            boundary_ids.add(sr["boundary_id"])

    # If any boundary_ids are present, they must all be the same
    if len(boundary_ids) > 1:
        return False

    seqs = [int(r["seq"]) for r in seq_records]
    counts = [int(r["running_count"]) for r in seq_records]

    expected = max(max(seqs) + 1, max(counts), sealed_total)

    # Check for duplicates
    duplicates = [s for s, n in Counter(seqs).items() if n > 1]
    if duplicates:
        return False

    # Check for gaps
    missing = sorted(set(range(expected)) - set(seqs))
    if missing:
        return False

    # Check running_count consistency (must == seq + 1)
    count_mismatch = [
        r for r in seq_records if int(r["running_count"]) != int(r["seq"]) + 1
    ]
    if count_mismatch:
        return False

    # Check record count matches expected
    if len(seq_records) != expected:
        return False

    return True
