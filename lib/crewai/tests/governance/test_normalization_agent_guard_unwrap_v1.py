"""agent-guard-unwrap-v1 conformance vectors for the GovernanceDecision contract.

Follow-up to #6030. The authoritative surface->normalized wrapper table lives in
the annex (linked in the PR thread); the `_strip` helper below is a faithful SUBSET
covering only the wrappers used by these two vectors, so `converge` is exercised
rather than asserted tautologically.
"""

from __future__ import annotations

import hashlib
import json
import re

from crewai.governance.governance_decision import GovernanceDecision

_NAME_VALUE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _params_hash(normalized_argv: list[str]) -> str:
    # For an array of ASCII strings, JCS == compact, non-escaping json.dumps,
    # byte-identical to the annex's rfc8785 output; no rfc8785 dep needed here.
    jcs = json.dumps(normalized_argv, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(jcs.encode("utf-8")).hexdigest()


def _strip(argv: list[str]) -> list[str]:
    """Reduce transparent wrappers / env-prefixes to the real executable+argv.

    Faithful SUBSET of agent-guard-unwrap-v1.md (sudo/doas/env/nice/nohup/timeout
    + NAME=value prefixes). The annex table is authoritative and covers more.
    """
    i, n = 0, len(argv)
    while i < n:
        tok = argv[i]
        if _NAME_VALUE.match(tok):                 # FOO=1 prefix
            i += 1
        elif tok in ("sudo", "doas"):
            i += 1
            while i < n and argv[i].startswith("-"):
                i += 2 if argv[i] == "-u" else 1   # -u consumes a value
        elif tok == "env":
            i += 1
            while i < n and _NAME_VALUE.match(argv[i]):
                i += 1
        elif tok == "nice":
            i += 1
            if i < n and argv[i] == "-n":
                i += 2
        elif tok == "nohup":
            i += 1
        elif tok == "timeout":
            i += 1
            while i < n and argv[i].startswith("-"):
                i += 2 if argv[i] == "-s" else 1   # -s consumes a signal
            if i < n:                              # DURATION
                i += 1
        else:
            break
    return argv[i:]


# --- Vector 1: CONVERGE -------------------------------------------------------
CONVERGE_CANONICAL = ["rm", "-rf", "/tmp/build"]
CONVERGE_SURFACE_FORMS = [
    ["rm", "-rf", "/tmp/build"],
    ["sudo", "rm", "-rf", "/tmp/build"],
    ["env", "FOO=1", "rm", "-rf", "/tmp/build"],
    ["FOO=1", "rm", "-rf", "/tmp/build"],
    ["timeout", "5", "rm", "-rf", "/tmp/build"],
    ["timeout", "-s", "KILL", "30", "rm", "-rf", "/tmp/build"],
    ["nice", "-n", "10", "nohup", "rm", "-rf", "/tmp/build"],
    ["sudo", "-u", "root", "env", "BAR=2", "rm", "-rf", "/tmp/build"],
]

FIXTURE_CONVERGE: GovernanceDecision = {
    "decision_id": "d-agunwrap-001",
    "intent_ref": "sha256:" + "a" * 64,
    "receipt_ref": "sha256:" + "b" * 64,
    "agent_id": "ops-agent",
    "agent_role": "Ops",
    "tool": "run_shell",
    "request_id": "req-agunwrap-001",
    "params_hash": _params_hash(CONVERGE_CANONICAL),
    "normalized_scope": "shell",
    "normalization_id": "agent-guard-unwrap-v1",
    "idempotency_key": "idem-agunwrap-001",
    "target_state_digest": None,
    "policy_refs": ["deny-destructive-v1"],
    "decision": "deny",
    "reason": "destructive rm -rf reached through transparent wrappers",
    "issued_at": "2026-07-05T00:00:00Z",
    "seq": 0,
    "running_count": 1,
}

# --- Vector 2: DIVERGE (same wrapper, different real executable) ---------------
DIVERGE_SURFACE_A = ["timeout", "5", "rm", "-rf", "/tmp/x"]
DIVERGE_SURFACE_B = ["timeout", "5", "ls", "/tmp/x"]


def test_converge_forms_recompute_to_one_params_hash() -> None:
    expected = _params_hash(CONVERGE_CANONICAL)
    assert FIXTURE_CONVERGE["params_hash"] == expected          # recompute, don't trust
    for form in CONVERGE_SURFACE_FORMS:
        assert _strip(form) == CONVERGE_CANONICAL, form          # wrappers are transparent
        assert _params_hash(_strip(form)) == expected, form


def test_diverge_pair_has_distinct_params_hash() -> None:
    assert _strip(DIVERGE_SURFACE_A) == ["rm", "-rf", "/tmp/x"]
    assert _strip(DIVERGE_SURFACE_B) == ["ls", "/tmp/x"]
    assert _params_hash(_strip(DIVERGE_SURFACE_A)) != _params_hash(_strip(DIVERGE_SURFACE_B))


def test_params_hash_is_recomputed_not_trusted() -> None:
    forged = dict(FIXTURE_CONVERGE, params_hash="sha256:" + "0" * 64)
    assert forged["params_hash"] != _params_hash(CONVERGE_CANONICAL)


def test_normalization_id_selects_the_annex_rule() -> None:
    assert FIXTURE_CONVERGE["normalization_id"] == "agent-guard-unwrap-v1"
