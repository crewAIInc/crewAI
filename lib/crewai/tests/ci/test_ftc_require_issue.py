"""Regression tests for the first-time contributor issue-gate workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

WORKFLOW_PATH = Path(__file__).resolve().parents[4] / (
    ".github/workflows/ftc-require-issue.yml"
)


def _embedded_python() -> str:
    text = WORKFLOW_PATH.read_text()
    start = text.index("python3 << 'PY'\n") + len("python3 << 'PY'\n")
    end = text.rindex("\n          PY\n")
    indent = "          "
    return "\n".join(
        line[len(indent) :] if line.startswith(indent) else line
        for line in text[start:end].splitlines()
    )


def _run_gate(
    *,
    title: str,
    body: str,
    issue_payloads: dict[int, dict],
    author_permission: str | None = None,
    actor_permission: str | None = None,
    event_action: str = "opened",
    expect_exit: int | None = 0,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_check_output(args: list[str], **_kwargs: object) -> str:
        if args[:2] == ["gh", "pr"] and "view" in args:
            return json.dumps({"title": title, "body": body, "state": "OPEN"})
        raise AssertionError(f"unexpected check_output: {args}")

    def fake_run(args: list[str], **_kwargs: object) -> mock.Mock:
        calls.append(list(args))
        result = mock.Mock()
        result.returncode = 0
        result.stderr = ""
        result.stdout = ""
        if args[:2] == ["gh", "api"] and "/collaborators/" in args[2]:
            permission = (
                author_permission
                if args[2].endswith("/maintainer/permission")
                else actor_permission
            )
            if permission is None:
                result.returncode = 1
                result.stderr = "404 Not Found"
            else:
                result.stdout = json.dumps({"permission": permission})
            return result
        if args[:2] == ["gh", "api"] and "/issues/" in args[2]:
            number = int(args[2].rsplit("/", 1)[-1])
            result.stdout = json.dumps(issue_payloads[number])
        return result

    env = {
        "REPO": "crewAIInc/crewAI",
        "PR_NUMBER": "99",
        "AUTHOR_LOGIN": "maintainer",
        "ACTOR_LOGIN": "org-member",
        "EVENT_ACTION": event_action,
        "AUTHOR_ASSOCIATION": "FIRST_TIME_CONTRIBUTOR",
    }
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch("subprocess.check_output", side_effect=fake_check_output),
        mock.patch("subprocess.run", side_effect=fake_run),
    ):
        compiled = compile(_embedded_python(), "<workflow>", "exec")
        if expect_exit is None:
            exec(compiled, {})  # noqa: S102
        else:
            with pytest.raises(SystemExit) as exited:
                exec(compiled, {})  # noqa: S102
            assert exited.value.code == expect_exit
    return calls


@pytest.mark.parametrize(
    "body",
    [
        "Related to #123",
        "crewAIInc/crewAI#123",
        "https://github.com/crewAIInc/crewAI/issues/123",
    ],
)
def test_open_issue_mention_blocks_close(body: str) -> None:
    calls = _run_gate(
        title="feat: example",
        body=body,
        issue_payloads={123: {"state": "open"}},
    )

    assert not any(call[:3] == ["gh", "pr", "close"] for call in calls)
    assert not any(
        call[:3] == ["gh", "pr", "edit"] and "--add-label" in call for call in calls
    )
    assert any(
        call[:2] == ["gh", "api"] and call[2].endswith("/issues/123") for call in calls
    )


def test_foreign_repo_reference_closes_pr() -> None:
    calls = _run_gate(
        title="feat: example",
        body="other/repo#123",
        issue_payloads={123: {"state": "open"}},
        expect_exit=None,
    )

    assert any(call[:3] == ["gh", "pr", "close"] for call in calls)
    assert any(
        call[:3] == ["gh", "pr", "edit"]
        and call[call.index("--add-label") + 1] == "needs-issue"
        for call in calls
        if "--add-label" in call
    )
    assert not any(
        call[:2] == ["gh", "api"] and "/issues/" in call[2] for call in calls
    )


@pytest.mark.parametrize("permission", ["write", "push", "maintain", "admin"])
def test_repository_maintainer_bypasses_issue_gate(permission: str) -> None:
    calls = _run_gate(
        title="feat: example",
        body="",
        issue_payloads={},
        author_permission=permission,
    )

    assert not any(call[:3] == ["gh", "pr", "close"] for call in calls)
    assert not any(
        call[:2] == ["gh", "api"] and "/issues/" in call[2] for call in calls
    )


def test_read_only_contributor_still_requires_issue() -> None:
    calls = _run_gate(
        title="feat: example",
        body="",
        issue_payloads={},
        author_permission="read",
        expect_exit=None,
    )

    assert any(call[:3] == ["gh", "pr", "close"] for call in calls)


def test_organization_member_reopening_pr_bypasses_issue_gate() -> None:
    calls = _run_gate(
        title="feat: example",
        body="",
        issue_payloads={},
        actor_permission="push",
        event_action="reopened",
    )

    assert not any(call[:3] == ["gh", "pr", "close"] for call in calls)
    assert not any(
        call[:2] == ["gh", "api"] and "/issues/" in call[2] for call in calls
    )
