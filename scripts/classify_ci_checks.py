"""Deterministic CI check-state classifier for CrewAI PR triage.

Normalizes raw GitHub CI check data into a deterministic JSON contract
so that cross-repo planning and execution can rely on a stable,
machine-readable CI-state output.

Categories
----------
- ``passed``         -- every check completed successfully
- ``failed``         -- at least one check failed, timed-out, or was cancelled
- ``pending``        -- at least one check is still queued or in progress
- ``no_checks``      -- the PR has no associated check runs or commit statuses
- ``policy_blocked`` -- at least one check requires manual action (e.g. review)

Usage
-----
Pipe JSON from the GitHub Checks API (or a compatible payload) into stdin::

    gh api repos/{owner}/{repo}/commits/{ref}/check-runs | python scripts/classify_ci_checks.py

Or supply a file path as the first positional argument::

    python scripts/classify_ci_checks.py checks.json

The script prints a single JSON object to stdout and exits with code 0 for
``passed``, 1 for ``failed``/``policy_blocked``, and 2 for ``pending``/``no_checks``.

Example output::

    {
      "state": "failed",
      "total": 3,
      "summary": "1 failed, 2 passed (3 total)",
      "checks": [
        {
          "name": "tests (3.12)",
          "status": "completed",
          "conclusion": "failure",
          "started_at": "2026-02-24T10:00:00Z",
          "completed_at": "2026-02-24T10:05:00Z"
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Public state constants
# ---------------------------------------------------------------------------

PASSED: str = "passed"
FAILED: str = "failed"
PENDING: str = "pending"
NO_CHECKS: str = "no_checks"
POLICY_BLOCKED: str = "policy_blocked"

ALL_STATES: frozenset[str] = frozenset(
    {PASSED, FAILED, PENDING, NO_CHECKS, POLICY_BLOCKED}
)

# ---------------------------------------------------------------------------
# Internal mapping helpers
# ---------------------------------------------------------------------------

# GitHub check-run conclusions that map to *failed*
_FAILED_CONCLUSIONS: frozenset[str] = frozenset(
    {"failure", "timed_out", "cancelled", "startup_failure"}
)

# GitHub check-run conclusions that map to *policy_blocked*
_POLICY_CONCLUSIONS: frozenset[str] = frozenset({"action_required"})

# GitHub check-run statuses that map to *pending*
_PENDING_STATUSES: frozenset[str] = frozenset({"queued", "in_progress", "waiting"})

# GitHub commit-status states that map to *failed*
_FAILED_COMMIT_STATES: frozenset[str] = frozenset({"failure", "error"})

# GitHub commit-status states that map to *pending*
_PENDING_COMMIT_STATES: frozenset[str] = frozenset({"pending"})


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def _extract_check_metadata(check: dict[str, Any]) -> dict[str, Any]:
    """Extract audit-relevant metadata from a single check run or status.

    Parameters
    ----------
    check:
        A single check-run or commit-status object from the GitHub API.

    Returns
    -------
    dict:
        Normalized metadata dict with name, status, conclusion, and timestamps.
    """
    return {
        "name": check.get("name") or check.get("context") or "",
        "status": check.get("status") or check.get("state") or "",
        "conclusion": check.get("conclusion") or "",
        "started_at": check.get("started_at") or "",
        "completed_at": check.get("completed_at") or check.get("updated_at") or "",
    }


def classify(payload: dict[str, Any]) -> dict[str, Any]:
    """Classify CI check data into a deterministic state.

    Accepts the JSON body returned by the GitHub ``check-runs`` endpoint
    (which wraps runs in ``{"total_count": N, "check_runs": [...]}``),
    a plain list of check-run objects, or a combined payload that also
    includes commit statuses under the ``statuses`` key.

    Parameters
    ----------
    payload:
        Raw GitHub API response or a dict with ``check_runs`` and/or
        ``statuses`` lists.

    Returns
    -------
    dict:
        Deterministic JSON-serialisable result with keys ``state``,
        ``total``, ``summary``, and ``checks`` (source metadata).

    Examples
    --------
    >>> result = classify({"check_runs": [], "statuses": []})
    >>> result["state"]
    'no_checks'

    >>> result = classify({
    ...     "check_runs": [
    ...         {"name": "lint", "status": "completed", "conclusion": "success"}
    ...     ]
    ... })
    >>> result["state"]
    'passed'
    """
    # Normalise input: accept top-level list or wrapped object
    if isinstance(payload.get("check_runs"), list):
        check_runs: list[dict[str, Any]] = payload["check_runs"]
    elif isinstance(payload, list):  # type: ignore[arg-type]
        check_runs = payload  # type: ignore[assignment]
    else:
        check_runs = []

    statuses: list[dict[str, Any]] = payload.get("statuses", []) if isinstance(payload, dict) else []

    all_metadata: list[dict[str, Any]] = []
    has_policy_blocked = False
    has_failed = False
    has_pending = False

    # --- Classify check runs ---
    for cr in check_runs:
        all_metadata.append(_extract_check_metadata(cr))
        status = (cr.get("status") or "").lower()
        conclusion = (cr.get("conclusion") or "").lower()

        if conclusion in _POLICY_CONCLUSIONS:
            has_policy_blocked = True
        elif conclusion in _FAILED_CONCLUSIONS:
            has_failed = True
        elif status in _PENDING_STATUSES:
            has_pending = True
        # completed + success/neutral/skipped/stale → not a problem

    # --- Classify commit statuses ---
    for cs in statuses:
        all_metadata.append(_extract_check_metadata(cs))
        state = (cs.get("state") or "").lower()

        if state in _FAILED_COMMIT_STATES:
            has_failed = True
        elif state in _PENDING_COMMIT_STATES:
            has_pending = True

    # --- Determine aggregate state (priority order) ---
    total = len(all_metadata)

    if total == 0:
        state = NO_CHECKS
    elif has_policy_blocked:
        state = POLICY_BLOCKED
    elif has_failed:
        state = FAILED
    elif has_pending:
        state = PENDING
    else:
        state = PASSED

    return {
        "state": state,
        "total": total,
        "summary": _build_summary(state, all_metadata),
        "checks": all_metadata,
    }


def _build_summary(state: str, checks: list[dict[str, Any]]) -> str:
    """Build a human-readable one-line summary.

    Parameters
    ----------
    state:
        The classified state string.
    checks:
        List of normalized check metadata dicts.

    Returns
    -------
    str:
        Human-readable summary string.
    """
    total = len(checks)
    if total == 0:
        return "No CI checks found"

    # Count by conclusion/status bucket
    counts: dict[str, int] = {}
    for c in checks:
        conclusion = c.get("conclusion", "")
        status = c.get("status", "")
        # Use conclusion if available, otherwise status
        bucket = conclusion if conclusion else status
        if not bucket:
            bucket = "unknown"
        counts[bucket] = counts.get(bucket, 0) + 1

    parts = [f"{v} {k}" for k, v in sorted(counts.items())]
    return f"{', '.join(parts)} ({total} total)"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

# Exit codes aligned to state severity
_EXIT_CODES: dict[str, int] = {
    PASSED: 0,
    FAILED: 1,
    POLICY_BLOCKED: 1,
    PENDING: 2,
    NO_CHECKS: 2,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point: read JSON from *stdin* or a file and classify.

    Parameters
    ----------
    argv:
        Command-line arguments (default: ``sys.argv[1:]``).

    Returns
    -------
    int:
        Exit code (0 = passed, 1 = failed/blocked, 2 = pending/no checks).
    """
    args = argv if argv is not None else sys.argv[1:]

    try:
        if args:
            with open(args[0]) as fh:
                raw = fh.read()
        else:
            raw = sys.stdin.read()

        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)  # noqa: T201
        return 1

    result = classify(payload)
    print(json.dumps(result, indent=2))  # noqa: T201
    return _EXIT_CODES.get(result["state"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
