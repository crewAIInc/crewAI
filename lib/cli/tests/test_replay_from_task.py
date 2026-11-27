"""Exit-code tests for the replay CLI subprocess."""

from __future__ import annotations

import subprocess

from click.testing import CliRunner

from crewai_cli.cli import replay


def _patch_replay_environment(monkeypatch) -> None:
    monkeypatch.setattr("crewai_cli.cli.get_or_create_project_id", lambda: None)
    monkeypatch.setattr(
        "crewai_cli.replay_from_task.build_env_with_all_tool_credentials",
        lambda: {},
    )


def test_replay_cli_propagates_child_exit_code(monkeypatch) -> None:
    _patch_replay_environment(monkeypatch)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(
            7,
            ["uv", "run", "replay", "task-1"],
            output="replay failed",
        )

    monkeypatch.setattr("crewai_cli.replay_from_task.subprocess.run", fail)

    result = CliRunner().invoke(replay, ["--task-id", "task-1"])

    assert result.exit_code == 7
    assert "returned non-zero exit status 7" in result.output


def test_replay_cli_exits_one_on_unexpected_error(monkeypatch) -> None:
    _patch_replay_environment(monkeypatch)

    def fail(*args, **kwargs):
        raise FileNotFoundError("uv is not installed")

    monkeypatch.setattr("crewai_cli.replay_from_task.subprocess.run", fail)

    result = CliRunner().invoke(replay, ["--task-id", "task-1"])

    assert result.exit_code == 1
    assert "uv is not installed" in result.output
