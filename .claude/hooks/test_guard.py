"""Behavior tests for the contributor-rule guard hook.

Run with: uv run pytest .claude/hooks/test_guard.py -q

The module is loaded by path rather than imported by name: `.claude/hooks` is not
a package and must not be added to `sys.path`, which would leak into other tests.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest


def _load_guard() -> ModuleType:
    path = Path(__file__).parent / "guard.py"
    spec = importlib.util.spec_from_file_location("_guard_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


BLOCKED_COMMANDS: list[tuple[str, str]] = [
    ("pip install ruff", "do not use pip directly"),
    ("pip3 install ruff", "do not use pip directly"),
    ("sudo pip install ruff", "do not use pip directly"),
    ("python -m pip install ruff", "do not use pip directly"),
    ("uv sync && pip install ruff", "do not use pip directly"),
    ("pip uninstall crewai", "do not use pip directly"),
    ("git commit -m 'x' --no-verify", "do not use --no-verify"),
    ("git commit -n -m 'x'", "do not use --no-verify"),
    ("git push --no-verify", "do not use --no-verify"),
    ("git -c user.name=x commit --no-verify -m 'x'", "do not use --no-verify"),
    ("uv sync && git commit --no-verify -m 'x'", "do not use --no-verify"),
    ("rm docs/images/flow.png", "docs/images/"),
    ("mv docs/images/a.png docs/images/b.png", "docs/images/"),
    ("rm -rf docs/v1.15.0", "frozen release snapshots"),
    # Version-suffixed interpreters must not bypass the pip rule.
    ("pip3.12 install ruff", "do not use pip directly"),
    ("pip2 install ruff", "do not use pip directly"),
    ("python3.12 -m pip install ruff", "do not use pip directly"),
    # The protected directory named without a trailing slash.
    ("rm -rf docs/images", "docs/images/"),
    ("mv docs/images docs/img", "docs/images/"),
    # Writes into a frozen snapshot by redirection or copy, not just rm/mv.
    ("echo x > docs/v1.15.0/index.mdx", "frozen release snapshots"),
    ("cat tmp.mdx >> docs/v1.15.0/index.mdx", "frozen release snapshots"),
    ("cp new.mdx docs/v1.15.0/index.mdx", "frozen release snapshots"),
    ("sed -i 's/a/b/' docs/v1.15.0/index.mdx", "frozen release snapshots"),
    ("tee docs/v1.15.0/index.mdx < new.mdx", "frozen release snapshots"),
]

ALLOWED_COMMANDS: list[str] = [
    "uv add --package crewai httpx",
    "uv add --dev pytest",
    "uv sync --all-groups --all-extras",
    "uv pip install -e .",
    "uv run pytest lib/crewai/tests -x -q",
    "uv run pytest -n auto --dist=loadfile lib/crewai/tests",
    "git commit -m 'feat(agents): add skill loader'",
    "git push origin main",
    'grep -rn "pip install" .github/CONTRIBUTING.md',
    "echo 'never use --no-verify' >> notes.txt",
    'grep -rn "no-verify" .github/CONTRIBUTING.md',
    "uv run pytest -n auto lib/crewai/tests",
    "uv run pip-audit --skip-editable --ignore-vuln PYSEC-2024-277",
    "rm docs/edge/en/scratch.mdx",
    # Read-only access to a frozen snapshot is allowed; only writes are blocked.
    "cat docs/v1.15.0/index.mdx",
    "grep -rn 'agents' docs/v1.15.0/ > /tmp/hits.txt",
    "less docs/v1.15.0/index.mdx",
    "ls docs/images/",
    # `-n` on push is --dry-run, not skip-hooks.
    "git push -n origin main",
    "git push --dry-run origin main",
    # A protected path named in a later, unrelated command segment.
    "rm /tmp/scratch.txt && ls docs/images/",
    "rm /tmp/scratch.txt && cat docs/v1.15.0/index.mdx",
    # Paths that merely start with the same prefix.
    "rm docs/imagesets/old.png",
]


@pytest.mark.parametrize(("command", "expected"), BLOCKED_COMMANDS)
def test_blocked_commands_report_the_rule_they_violate(
    command: str, expected: str
) -> None:
    reason = guard.bash_violation(command)
    assert reason is not None, f"expected {command!r} to be blocked"
    assert expected in reason


@pytest.mark.parametrize("command", ALLOWED_COMMANDS)
def test_allowed_commands_pass_through(command: str) -> None:
    assert guard.bash_violation(command) is None


def test_blocked_reasons_cite_a_committed_document() -> None:
    for command, _ in BLOCKED_COMMANDS:
        reason = guard.bash_violation(command)
        assert reason is not None
        assert "CONTRIBUTING.md" in reason or "AGENTS.md" in reason


HEREDOC_COMMIT = """git commit -m "$(cat <<'EOF'
fix(agents): close guard bypasses found in review

- rm -rf docs/images without a trailing slash was not matched
- pip install was reachable via pip3.12
EOF
)"
"""


def test_heredoc_bodies_are_data_not_commands() -> None:
    """A commit message discussing a blocked command must not itself be blocked."""
    assert guard.bash_violation(HEREDOC_COMMIT) is None


def test_a_heredoc_writing_into_a_frozen_snapshot_is_still_blocked() -> None:
    """Stripping the body must not hide a redirection on the command line."""
    command = "cat > docs/v1.15.0/index.mdx <<'EOF'\nsome new content\nEOF\n"
    reason = guard.bash_violation(command)
    assert reason is not None
    assert "frozen release snapshots" in reason


def test_a_redirect_after_the_heredoc_delimiter_is_still_seen() -> None:
    """The opener's own line must survive stripping, redirect included."""
    command = "cat <<'EOF' > docs/v1.15.0/index.mdx\nsome new content\nEOF\n"
    reason = guard.bash_violation(command)
    assert reason is not None
    assert "frozen release snapshots" in reason


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("bash <<'EOF'\nrm -rf docs/images/a.png\nEOF\n", "docs/images/"),
        ("sh <<'EOF'\npip install ruff\nEOF\n", "do not use pip directly"),
        ("bash -s <<'EOF'\ngit commit --no-verify -m x\nEOF\n", "--no-verify"),
    ],
)
def test_a_heredoc_piped_into_a_shell_is_executable_not_data(
    command: str, expected: str
) -> None:
    """A body a shell will run must still be matched, not stripped as data."""
    reason = guard.bash_violation(command)
    assert reason is not None, f"expected {command!r} to be blocked"
    assert expected in reason


def test_a_command_after_a_heredoc_is_still_inspected() -> None:
    command = "cat <<'EOF' > notes.txt\njust notes\nEOF\npip install ruff\n"
    reason = guard.bash_violation(command)
    assert reason is not None
    assert "do not use pip directly" in reason


def test_override_marker_allows_a_stated_exception() -> None:
    command = "pip install vendored.whl  # policy-override: offline wheel, no index"
    assert guard.bash_violation(command) is None


def test_override_marker_does_not_leak_to_other_commands() -> None:
    assert guard.bash_violation("pip install ruff") is not None


@pytest.mark.parametrize(
    "path",
    [
        "docs/v1.15.0/index.mdx",
        "/repo/docs/v1.15.0/concepts/agents.mdx",
        "/repo/docs/v2/index.mdx",
    ],
)
def test_frozen_snapshot_paths_are_blocked(path: str) -> None:
    assert guard.edits_frozen_docs(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "docs/edge/en/index.mdx",
        "docs/edge/pt-BR/index.mdx",
        "lib/crewai/src/crewai/agent.py",
        "docs/versioning.md",
        "",
    ],
)
def test_editable_paths_are_allowed(path: str) -> None:
    assert guard.edits_frozen_docs(path) is False


def _run_main(event: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any] | None:
    """Drive main() with a hook event, returning the parsed decision if any."""
    payload = event if isinstance(event, str) else json.dumps(event)
    monkeypatch.setattr(sys, "stdin", io.StringIO(payload))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    guard.main()
    written = out.getvalue().strip()
    if not written:
        return None
    decision: dict[str, Any] = json.loads(written)
    return decision


def test_main_denies_a_violating_bash_call(monkeypatch: pytest.MonkeyPatch) -> None:
    decision = _run_main({"tool_input": {"command": "pip install ruff"}}, monkeypatch)
    assert decision is not None
    output = decision["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert "do not use pip directly" in output["permissionDecisionReason"]


def test_main_denies_a_write_to_a_frozen_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = _run_main(
        {"tool_input": {"file_path": "/repo/docs/v1.15.0/index.mdx"}}, monkeypatch
    )
    assert decision is not None
    reason = decision["hookSpecificOutput"]["permissionDecisionReason"]
    assert "frozen release snapshots" in reason


def test_main_stays_silent_on_an_allowed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main({"tool_input": {"command": "uv sync"}}, monkeypatch) is None


@pytest.mark.parametrize(
    "event",
    ["", "not json", "[]", '"a string"', "{}", '{"tool_input": null}'],
)
def test_malformed_events_never_block(
    event: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_main(event, monkeypatch) is None


def test_a_bash_call_is_not_evaluated_as_a_file_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command mentioning docs/v* is judged by the Bash rules, not the path rule."""
    assert (
        _run_main({"tool_input": {"command": "cat docs/v1.15.0/x.mdx"}}, monkeypatch)
        is None
    )
