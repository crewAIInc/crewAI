import subprocess
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from crewai_cli.cli import train
from crewai_cli.train_crew import train_crew


@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_crew_positive_iterations(mock_subprocess_run):
    n_iterations = 5
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "train", str(n_iterations)],
        returncode=0,
        stdout="Success",
        stderr="",
    )

    train_crew(n_iterations, "trained_agents_data.pkl")

    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "train", str(n_iterations), "trained_agents_data.pkl"],
        capture_output=False,
        text=True,
        check=True,
    )


@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_crew_zero_iterations(mock_subprocess_run):
    with pytest.raises(
        click.ClickException,
        match="The number of iterations must be a positive integer.",
    ):
        train_crew(0, "trained_agents_data.pkl")
    mock_subprocess_run.assert_not_called()


@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_crew_negative_iterations(mock_subprocess_run):
    with pytest.raises(
        click.ClickException,
        match="The number of iterations must be a positive integer.",
    ):
        train_crew(-2, "trained_agents_data.pkl")
    mock_subprocess_run.assert_not_called()


@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_crew_invalid_filename(mock_subprocess_run):
    with pytest.raises(
        click.ClickException, match=r"The filename must end with \.pkl"
    ):
        train_crew(5, "myfile.txt")
    mock_subprocess_run.assert_not_called()


@mock.patch("crewai_cli.train_crew.click")
@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_crew_called_process_error(mock_subprocess_run, click):
    n_iterations = 5
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["uv", "run", "train", str(n_iterations)],
        output="Error",
        stderr="Some error occurred",
    )
    with pytest.raises(SystemExit) as exc_info:
        train_crew(n_iterations, "trained_agents_data.pkl")

    assert exc_info.value.code == 1
    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "train", str(n_iterations), "trained_agents_data.pkl"],
        capture_output=False,
        text=True,
        check=True,
    )
    click.echo.assert_has_calls(
        [
            mock.call.echo(
                "An error occurred while training the crew: Command '['uv', 'run', 'train', '5']' returned non-zero exit status 1.",
                err=True,
            ),
            mock.call.echo("Error", err=True),
        ]
    )


@mock.patch("crewai_cli.train_crew.click")
@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_crew_unexpected_exception(mock_subprocess_run, click):
    n_iterations = 5
    mock_subprocess_run.side_effect = Exception("Unexpected error")
    with pytest.raises(SystemExit) as exc_info:
        train_crew(n_iterations, "trained_agents_data.pkl")

    assert exc_info.value.code == 1
    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "train", str(n_iterations), "trained_agents_data.pkl"],
        capture_output=False,
        text=True,
        check=True,
    )
    click.echo.assert_called_once_with(
        "An unexpected error occurred: Unexpected error", err=True
    )


@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_crew_exits_with_the_script_exit_code(mock_subprocess_run):
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=42, cmd=["uv", "run", "train", "5"]
    )

    with pytest.raises(SystemExit) as exc_info:
        train_crew(5, "trained_agents_data.pkl")

    assert exc_info.value.code == 42


@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_command_exits_with_the_script_exit_code(
    mock_subprocess_run, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=42, cmd=["uv", "run", "train", "1"]
    )

    result = CliRunner().invoke(train, ["-n", "1"])

    assert result.exit_code == 42
    assert "An error occurred while training the crew" in result.output


@mock.patch("crewai_cli.train_crew.subprocess.run")
def test_train_command_exits_zero_when_the_script_succeeds(
    mock_subprocess_run, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "train", "1"], returncode=0, stdout="", stderr=""
    )

    result = CliRunner().invoke(train, ["-n", "1"])

    assert result.exit_code == 0
