"""Phase 0 of ``devtools release`` opens a catalog PR only when the catalog changed."""

from pathlib import Path

from crewai_devtools import cli
import pytest


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    """Record git/gh commands; ``git status --porcelain`` output is set per test."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(cli, "update_catalog", lambda root: root / cli.CATALOG_RELATIVE)
    monkeypatch.setattr(
        cli,
        "create_or_reset_branch",
        lambda branch: recorded.append(["branch", branch]),
    )
    return recorded


def _fake_run(recorded: list[list[str]], status: str):
    def run(cmd: list[str], cwd: Path | None = None) -> str:
        recorded.append(cmd)
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return status
        if cmd[:3] == ["gh", "pr", "create"]:
            return "https://github.com/o/r/pull/1"
        return ""

    return run


def test_unchanged_catalog_creates_no_branch_or_pr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[list[str]]
) -> None:
    monkeypatch.setattr(cli, "run_command", _fake_run(calls, status=""))

    assert cli._update_catalog_and_create_pr(tmp_path, "1.2.3", dry_run=False) is None

    assert not any(c[0] == "branch" for c in calls)
    assert not any(c[:2] == ["gh", "pr"] for c in calls)


def test_changed_catalog_opens_a_pr_from_its_own_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[list[str]]
) -> None:
    monkeypatch.setattr(cli, "run_command", _fake_run(calls, status=" M catalog.json"))

    branch = cli._update_catalog_and_create_pr(tmp_path, "1.2.3", dry_run=False)

    assert branch == "chore/update-litellm-catalog-v1.2.3"
    assert ["branch", branch] in calls
    pr = next(c for c in calls if c[:3] == ["gh", "pr", "create"])
    assert "chore: update litellm model catalog for v1.2.3" in pr


def test_declining_the_branch_reset_restores_the_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[list[str]]
) -> None:
    monkeypatch.setattr(cli, "run_command", _fake_run(calls, status=" M catalog.json"))

    def decline(branch: str) -> None:
        raise SystemExit(1)

    monkeypatch.setattr(cli, "create_or_reset_branch", decline)

    with pytest.raises(SystemExit):
        cli._update_catalog_and_create_pr(tmp_path, "1.2.3", dry_run=False)

    assert any(c[:3] == ["git", "checkout", "--"] for c in calls)
    assert not any(c[:2] == ["gh", "pr"] for c in calls)
