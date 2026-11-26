"""Replay subprocess failures must reach the CLI exit status."""

from subprocess import CalledProcessError
from unittest.mock import patch

from click.testing import CliRunner

from crewai_cli.cli import replay


def test_child_failure_propagates_exit_code():
    with patch(
        "crewai_cli.replay_from_task.subprocess.run",
        side_effect=CalledProcessError(7, ["uv", "run", "replay"]),
    ):
        result = CliRunner().invoke(replay, ["--task-id", "task-1"])

    assert result.exit_code == 7


def test_launch_failure_exits_nonzero():
    with patch(
        "crewai_cli.replay_from_task.subprocess.run", side_effect=FileNotFoundError("uv")
    ):
        result = CliRunner().invoke(replay, ["--task-id", "task-1"])

    assert result.exit_code == 1
