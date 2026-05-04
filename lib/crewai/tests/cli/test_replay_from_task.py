"""Tests for ``crewai replay`` and the trained-agents file plumbing."""

import subprocess
from unittest import mock

from click.testing import CliRunner
import pytest

from crewai.cli import replay_from_task
from crewai.cli.cli import replay


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@mock.patch("crewai.cli.cli.replay_task_command")
def test_replay_passes_filename(replay_task_command_mock: mock.Mock, runner: CliRunner) -> None:
    result = runner.invoke(replay, ["-t", "abc123", "-f", "my_custom.pkl"])

    replay_task_command_mock.assert_called_once_with(
        "abc123", trained_agents_file="my_custom.pkl"
    )
    assert result.exit_code == 0


@mock.patch("crewai.cli.cli.replay_task_command")
def test_replay_without_filename_passes_none(
    replay_task_command_mock: mock.Mock, runner: CliRunner
) -> None:
    result = runner.invoke(replay, ["-t", "abc123"])

    replay_task_command_mock.assert_called_once_with(
        "abc123", trained_agents_file=None
    )
    assert result.exit_code == 0


@mock.patch("crewai.cli.replay_from_task.subprocess.run")
def test_replay_task_command_sets_env_var(mock_subprocess_run: mock.Mock) -> None:
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "replay", "abc123"], returncode=0
    )
    replay_from_task.replay_task_command("abc123", trained_agents_file="my_custom.pkl")

    _, kwargs = mock_subprocess_run.call_args
    assert kwargs["env"]["CREWAI_TRAINED_AGENTS_FILE"] == "my_custom.pkl"


@mock.patch("crewai.cli.replay_from_task.subprocess.run")
def test_replay_task_command_omits_env_var_without_filename(
    mock_subprocess_run: mock.Mock,
) -> None:
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "replay", "abc123"], returncode=0
    )
    replay_from_task.replay_task_command("abc123")

    _, kwargs = mock_subprocess_run.call_args
    assert "CREWAI_TRAINED_AGENTS_FILE" not in kwargs["env"]