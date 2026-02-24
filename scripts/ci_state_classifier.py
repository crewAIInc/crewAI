#!/usr/bin/env python3
"""Classify GitHub PR checks into deterministic machine-readable states.

Outputs one of:
- passed
- failed
- pending
- no_checks
- policy_blocked
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAIL_VALUES = {
    "FAILURE",
    "ERROR",
    "TIMED_OUT",
    "CANCELLED",
    "ACTION_REQUIRED",
    "STARTUP_FAILURE",
    "STALE",
}
PENDING_VALUES = {"PENDING", "QUEUED", "IN_PROGRESS", "REQUESTED", "WAITING", "EXPECTED"}
POLICY_PATTERNS = [
    r"\bcla\b",
    r"license/cla",
    r"code[- ]?owners",
    r"dco",
    r"policy",
    r"compliance",
    r"signed[- ]off",
]


@dataclass
class Check:
    name: str
    conclusion: str
    status: str


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _to_checks(raw: list[dict[str, Any]]) -> list[Check]:
    checks: list[Check] = []
    for item in raw:
        name = (
            item.get("name")
            or item.get("context")
            or item.get("__typename")
            or "unknown"
        )
        conclusion = _norm(item.get("conclusion") or item.get("state"))
        status = _norm(item.get("status") or item.get("state"))
        checks.append(Check(name=str(name), conclusion=conclusion, status=status))
    return checks


def _is_policy(check: Check) -> bool:
    name = check.name.lower()
    return any(re.search(pattern, name) for pattern in POLICY_PATTERNS)


def classify(checks: list[Check]) -> str:
    if not checks:
        return "no_checks"

    failing = [c for c in checks if c.conclusion in FAIL_VALUES or c.status in FAIL_VALUES]
    pending = [c for c in checks if c.status in PENDING_VALUES or c.conclusion in PENDING_VALUES]

    if failing:
        non_policy_failures = [c for c in failing if not _is_policy(c)]
        if non_policy_failures:
            return "failed"

        non_policy_pending = [c for c in pending if not _is_policy(c)]
        if non_policy_pending:
            return "pending"

        return "policy_blocked"

    if pending:
        if all(_is_policy(c) for c in pending):
            return "policy_blocked"
        return "pending"

    return "passed"


def _load_status_rollup(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "statusCheckRollup" in payload:
        rollup = payload["statusCheckRollup"]
        if rollup is None:
            return []
        if isinstance(rollup, list):
            return rollup
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected JSON list or object with statusCheckRollup list")


def _fetch_status_rollup(repo: str, pr: int) -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            "--repo",
            repo,
            str(pr),
            "--json",
            "statusCheckRollup,url",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    return payload.get("statusCheckRollup", []) or []


def _report(repo: str, pr: int | None, checks: list[Check]) -> dict[str, Any]:
    return {
        "repo": repo,
        "pr": pr,
        "classification": classify(checks),
        "check_count": len(checks),
        "checks": [
            {
                "name": c.name,
                "conclusion": c.conclusion,
                "status": c.status,
                "is_policy_check": _is_policy(c),
            }
            for c in checks
        ],
    }


def _self_test() -> None:
    assert classify([]) == "no_checks"
    assert (
        classify([Check("unit", "SUCCESS", "COMPLETED")]) == "passed"
    )
    assert (
        classify([Check("build", "FAILURE", "COMPLETED")]) == "failed"
    )
    assert (
        classify([Check("build", "STARTUP_FAILURE", "COMPLETED")]) == "failed"
    )
    assert (
        classify([Check("license/cla", "", "QUEUED")]) == "policy_blocked"
    )
    assert (
        classify([Check("tests", "", "IN_PROGRESS")]) == "pending"
    )
    assert (
        classify([Check("required", "EXPECTED", "EXPECTED")]) == "pending"
    )
    assert (
        classify(
            [
                Check("license/cla", "ACTION_REQUIRED", "COMPLETED"),
                Check("tests", "", "IN_PROGRESS"),
            ]
        )
        == "pending"
    )
    print(json.dumps({"self_test": "ok"}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", help="owner/repo")
    parser.add_argument("--pr", type=int, help="PR number")
    parser.add_argument("--input", help="Path to JSON payload for statusCheckRollup")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0

    if args.input:
        raw = _load_status_rollup(Path(args.input))
        checks = _to_checks(raw)
        report = _report(args.repo or "fixture", args.pr, checks)
    else:
        if not args.repo or not args.pr:
            parser.error("--repo and --pr are required unless --input or --self-test is used")
        raw = _fetch_status_rollup(args.repo, args.pr)
        checks = _to_checks(raw)
        report = _report(args.repo, args.pr, checks)

    if args.pretty:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
