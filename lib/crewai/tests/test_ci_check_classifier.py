"""Tests for the deterministic CI check-state classifier.

Covers every category defined in the acceptance criteria for issue #4576:
  - passed
  - failed
  - pending
  - no_checks
  - policy_blocked

Also validates that source check metadata is retained for audit/review,
and that the output contract (JSON shape) is stable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Dynamic import of the classifier script from ``scripts/``
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "classify_ci_checks.py"

_spec = importlib.util.spec_from_file_location("classify_ci_checks", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

classify = _mod.classify
main = _mod.main
PASSED = _mod.PASSED
FAILED = _mod.FAILED
PENDING = _mod.PENDING
NO_CHECKS = _mod.NO_CHECKS
POLICY_BLOCKED = _mod.POLICY_BLOCKED
ALL_STATES = _mod.ALL_STATES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_check_run(
    name: str = "ci",
    status: str = "completed",
    conclusion: str = "success",
    started_at: str = "2026-01-01T00:00:00Z",
    completed_at: str = "2026-01-01T00:05:00Z",
) -> dict[str, Any]:
    """Build a minimal GitHub check-run dict."""
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _make_commit_status(
    context: str = "ci/status",
    state: str = "success",
    updated_at: str = "2026-01-01T00:05:00Z",
) -> dict[str, str]:
    """Build a minimal GitHub commit-status dict."""
    return {
        "context": context,
        "state": state,
        "updated_at": updated_at,
    }


# ===================================================================
# Category: no_checks
# ===================================================================


class TestNoChecks:
    """When there are zero check runs and zero statuses -> ``no_checks``."""

    def test_empty_check_runs_list(self) -> None:
        result = classify({"check_runs": []})
        assert result["state"] == NO_CHECKS
        assert result["total"] == 0

    def test_empty_payload(self) -> None:
        result = classify({})
        assert result["state"] == NO_CHECKS
        assert result["total"] == 0

    def test_empty_check_runs_and_statuses(self) -> None:
        result = classify({"check_runs": [], "statuses": []})
        assert result["state"] == NO_CHECKS
        assert result["total"] == 0

    def test_summary_message(self) -> None:
        result = classify({"check_runs": []})
        assert "No CI checks found" in result["summary"]


# ===================================================================
# Category: passed
# ===================================================================


class TestPassed:
    """All checks completed successfully -> ``passed``."""

    def test_single_success(self) -> None:
        result = classify({"check_runs": [_make_check_run()]})
        assert result["state"] == PASSED
        assert result["total"] == 1

    def test_multiple_successes(self) -> None:
        runs = [
            _make_check_run(name="lint"),
            _make_check_run(name="tests (3.10)"),
            _make_check_run(name="tests (3.12)"),
        ]
        result = classify({"check_runs": runs})
        assert result["state"] == PASSED
        assert result["total"] == 3

    def test_neutral_conclusion_counts_as_passed(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(conclusion="neutral")]}
        )
        assert result["state"] == PASSED

    def test_skipped_conclusion_counts_as_passed(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(conclusion="skipped")]}
        )
        assert result["state"] == PASSED

    def test_commit_status_success(self) -> None:
        result = classify(
            {"check_runs": [], "statuses": [_make_commit_status(state="success")]}
        )
        assert result["state"] == PASSED

    def test_mixed_check_runs_and_statuses_all_pass(self) -> None:
        result = classify({
            "check_runs": [_make_check_run(name="build")],
            "statuses": [_make_commit_status(context="deploy", state="success")],
        })
        assert result["state"] == PASSED
        assert result["total"] == 2


# ===================================================================
# Category: failed
# ===================================================================


class TestFailed:
    """At least one check failed -> ``failed``."""

    def test_single_failure(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(conclusion="failure")]}
        )
        assert result["state"] == FAILED

    def test_timed_out(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(conclusion="timed_out")]}
        )
        assert result["state"] == FAILED

    def test_cancelled(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(conclusion="cancelled")]}
        )
        assert result["state"] == FAILED

    def test_startup_failure(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(conclusion="startup_failure")]}
        )
        assert result["state"] == FAILED

    def test_failure_among_successes(self) -> None:
        runs = [
            _make_check_run(name="lint"),
            _make_check_run(name="tests", conclusion="failure"),
            _make_check_run(name="build"),
        ]
        result = classify({"check_runs": runs})
        assert result["state"] == FAILED
        assert result["total"] == 3

    def test_commit_status_failure(self) -> None:
        result = classify(
            {"check_runs": [], "statuses": [_make_commit_status(state="failure")]}
        )
        assert result["state"] == FAILED

    def test_commit_status_error(self) -> None:
        result = classify(
            {"check_runs": [], "statuses": [_make_commit_status(state="error")]}
        )
        assert result["state"] == FAILED

    def test_failed_overrides_pending(self) -> None:
        """Failed takes precedence over pending."""
        runs = [
            _make_check_run(name="lint", status="in_progress", conclusion=""),
            _make_check_run(name="tests", conclusion="failure"),
        ]
        result = classify({"check_runs": runs})
        assert result["state"] == FAILED


# ===================================================================
# Category: pending
# ===================================================================


class TestPending:
    """At least one check still in progress or queued -> ``pending``."""

    def test_queued(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(status="queued", conclusion="")]}
        )
        assert result["state"] == PENDING

    def test_in_progress(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(status="in_progress", conclusion="")]}
        )
        assert result["state"] == PENDING

    def test_waiting(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(status="waiting", conclusion="")]}
        )
        assert result["state"] == PENDING

    def test_pending_among_successes(self) -> None:
        runs = [
            _make_check_run(name="lint"),
            _make_check_run(name="tests", status="in_progress", conclusion=""),
        ]
        result = classify({"check_runs": runs})
        assert result["state"] == PENDING

    def test_commit_status_pending(self) -> None:
        result = classify(
            {"check_runs": [], "statuses": [_make_commit_status(state="pending")]}
        )
        assert result["state"] == PENDING


# ===================================================================
# Category: policy_blocked
# ===================================================================


class TestPolicyBlocked:
    """A check requires manual action -> ``policy_blocked``."""

    def test_action_required(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(conclusion="action_required")]}
        )
        assert result["state"] == POLICY_BLOCKED

    def test_policy_blocked_overrides_failed(self) -> None:
        """policy_blocked has highest priority after no_checks."""
        runs = [
            _make_check_run(name="lint", conclusion="failure"),
            _make_check_run(name="review", conclusion="action_required"),
        ]
        result = classify({"check_runs": runs})
        assert result["state"] == POLICY_BLOCKED

    def test_policy_blocked_overrides_pending(self) -> None:
        runs = [
            _make_check_run(name="build", status="in_progress", conclusion=""),
            _make_check_run(name="policy", conclusion="action_required"),
        ]
        result = classify({"check_runs": runs})
        assert result["state"] == POLICY_BLOCKED


# ===================================================================
# Output contract / metadata retention
# ===================================================================


class TestOutputContract:
    """The JSON output has a stable shape and retains source metadata."""

    def test_result_keys(self) -> None:
        result = classify({"check_runs": [_make_check_run()]})
        assert set(result.keys()) == {"state", "total", "summary", "checks"}

    def test_state_is_a_known_value(self) -> None:
        for conclusion in ("success", "failure", "action_required"):
            result = classify({"check_runs": [_make_check_run(conclusion=conclusion)]})
            assert result["state"] in ALL_STATES

    def test_check_metadata_retained(self) -> None:
        cr = _make_check_run(name="my-job", conclusion="success")
        result = classify({"check_runs": [cr]})
        meta = result["checks"][0]
        assert meta["name"] == "my-job"
        assert meta["status"] == "completed"
        assert meta["conclusion"] == "success"
        assert meta["started_at"] == "2026-01-01T00:00:00Z"
        assert meta["completed_at"] == "2026-01-01T00:05:00Z"

    def test_commit_status_metadata_retained(self) -> None:
        cs = _make_commit_status(context="ci/deploy", state="success")
        result = classify({"check_runs": [], "statuses": [cs]})
        meta = result["checks"][0]
        assert meta["name"] == "ci/deploy"
        assert meta["status"] == "success"

    def test_result_is_json_serialisable(self) -> None:
        result = classify({
            "check_runs": [_make_check_run()],
            "statuses": [_make_commit_status()],
        })
        roundtripped = json.loads(json.dumps(result))
        assert roundtripped == result

    def test_total_matches_checks_length(self) -> None:
        runs = [_make_check_run(name=f"job-{i}") for i in range(5)]
        result = classify({"check_runs": runs})
        assert result["total"] == len(result["checks"]) == 5


# ===================================================================
# CLI entry-point (main)
# ===================================================================


class TestCLI:
    """Test the ``main()`` function that wraps classify for CLI use."""

    def test_exit_code_passed(self, tmp_path: Path) -> None:
        payload = {"check_runs": [_make_check_run()]}
        f = tmp_path / "input.json"
        f.write_text(json.dumps(payload))
        assert main([str(f)]) == 0

    def test_exit_code_failed(self, tmp_path: Path) -> None:
        payload = {"check_runs": [_make_check_run(conclusion="failure")]}
        f = tmp_path / "input.json"
        f.write_text(json.dumps(payload))
        assert main([str(f)]) == 1

    def test_exit_code_pending(self, tmp_path: Path) -> None:
        payload = {"check_runs": [_make_check_run(status="queued", conclusion="")]}
        f = tmp_path / "input.json"
        f.write_text(json.dumps(payload))
        assert main([str(f)]) == 2

    def test_exit_code_no_checks(self, tmp_path: Path) -> None:
        payload = {"check_runs": []}
        f = tmp_path / "input.json"
        f.write_text(json.dumps(payload))
        assert main([str(f)]) == 2

    def test_exit_code_policy_blocked(self, tmp_path: Path) -> None:
        payload = {"check_runs": [_make_check_run(conclusion="action_required")]}
        f = tmp_path / "input.json"
        f.write_text(json.dumps(payload))
        assert main([str(f)]) == 1

    def test_invalid_json_returns_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("NOT JSON")
        assert main([str(f)]) == 1

    def test_missing_file_returns_error(self) -> None:
        assert main(["/nonexistent/path.json"]) == 1


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Boundary and edge-case scenarios."""

    def test_check_run_with_missing_fields(self) -> None:
        """Gracefully handles check runs that omit optional fields."""
        result = classify({"check_runs": [{"status": "completed", "conclusion": "success"}]})
        assert result["state"] == PASSED
        meta = result["checks"][0]
        assert meta["name"] == ""
        assert meta["started_at"] == ""

    def test_case_insensitive_conclusion(self) -> None:
        """Conclusion strings are normalised to lowercase."""
        result = classify(
            {"check_runs": [_make_check_run(conclusion="FAILURE")]}
        )
        assert result["state"] == FAILED

    def test_case_insensitive_status(self) -> None:
        result = classify(
            {"check_runs": [_make_check_run(status="IN_PROGRESS", conclusion="")]}
        )
        assert result["state"] == PENDING

    def test_stale_conclusion_is_not_failure(self) -> None:
        """``stale`` is a non-blocking conclusion."""
        result = classify(
            {"check_runs": [_make_check_run(conclusion="stale")]}
        )
        assert result["state"] == PASSED

    def test_large_number_of_checks(self) -> None:
        """Classifier handles many checks without error."""
        runs = [_make_check_run(name=f"job-{i}") for i in range(500)]
        result = classify({"check_runs": runs})
        assert result["state"] == PASSED
        assert result["total"] == 500

    def test_mixed_all_states(self) -> None:
        """When all state types are present, policy_blocked wins."""
        runs = [
            _make_check_run(name="pass", conclusion="success"),
            _make_check_run(name="fail", conclusion="failure"),
            _make_check_run(name="pend", status="queued", conclusion=""),
            _make_check_run(name="block", conclusion="action_required"),
        ]
        result = classify({"check_runs": runs})
        assert result["state"] == POLICY_BLOCKED
        assert result["total"] == 4
