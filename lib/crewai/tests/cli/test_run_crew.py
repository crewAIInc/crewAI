"""Tests for the ``crewai run`` command and its subprocess plumbing."""

from unittest import mock

from click.testing import CliRunner
import pytest

from crewai_cli.cli import run
from crewai_cli.run_crew import CrewType, execute_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@mock.patch("crewai_cli.cli.run_crew")
def test_run_passes_filename_to_run_crew(run_crew_mock: mock.Mock, runner: CliRunner) -> None:
    result = runner.invoke(run, ["-f", "my_custom_trained.pkl"])

    run_crew_mock.assert_called_once_with(trained_agents_file="my_custom_trained.pkl")
    assert result.exit_code == 0


@mock.patch("crewai_cli.cli.run_crew")
def test_run_without_filename_passes_none(run_crew_mock: mock.Mock, runner: CliRunner) -> None:
    result = runner.invoke(run)

    run_crew_mock.assert_called_once_with(trained_agents_file=None)
    assert result.exit_code == 0


@mock.patch("crewai_cli.run_crew.subprocess.run")
@mock.patch(
    "crewai_cli.run_crew.build_env_with_all_tool_credentials",
    return_value={"EXISTING": "value"},
)
def test_execute_command_sets_env_var_when_filename_provided(
    _build_env: mock.Mock, subprocess_run: mock.Mock
) -> None:
    execute_command(CrewType.STANDARD, trained_agents_file="my_custom_trained.pkl")

    _, kwargs = subprocess_run.call_args
    assert kwargs["env"]["CREWAI_TRAINED_AGENTS_FILE"] == "my_custom_trained.pkl"
    assert kwargs["env"]["EXISTING"] == "value"


@mock.patch("crewai_cli.run_crew.subprocess.run")
@mock.patch(
    "crewai_cli.run_crew.build_env_with_all_tool_credentials",
    return_value={"EXISTING": "value"},
)
def test_execute_command_omits_env_var_when_filename_absent(
    _build_env: mock.Mock, subprocess_run: mock.Mock
) -> None:
    execute_command(CrewType.STANDARD)

    _, kwargs = subprocess_run.call_args
    assert "CREWAI_TRAINED_AGENTS_FILE" not in kwargs["env"]