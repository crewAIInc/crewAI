import subprocess
from unittest import mock

import pytest
from click.testing import CliRunner

from crewai.cli.cli import run
from crewai.cli.run_crew import execute_command, CrewType


@pytest.fixture
def runner():
    return CliRunner()


@mock.patch("crewai.cli.run_crew.execute_command")
@mock.patch("crewai.cli.run_crew.read_toml")
@mock.patch("crewai.cli.run_crew.get_crewai_version")
def test_run_crew_without_active_flag(mock_version, mock_toml, mock_execute, runner):
    """Test that run command works without --active flag (default behavior)."""
    mock_version.return_value = "0.148.0"
    mock_toml.return_value = {"tool": {"crewai": {"type": "standard"}}}
    
    result = runner.invoke(run)
    
    assert result.exit_code == 0
    mock_execute.assert_called_once_with(CrewType.STANDARD, False)


@mock.patch("crewai.cli.run_crew.execute_command")
@mock.patch("crewai.cli.run_crew.read_toml")
@mock.patch("crewai.cli.run_crew.get_crewai_version")
def test_run_crew_with_active_flag(mock_version, mock_toml, mock_execute, runner):
    """Test that run command works with --active flag."""
    mock_version.return_value = "0.148.0"
    mock_toml.return_value = {"tool": {"crewai": {"type": "standard"}}}
    
    result = runner.invoke(run, ["--active"])
    
    assert result.exit_code == 0
    mock_execute.assert_called_once_with(CrewType.STANDARD, True)


@mock.patch("crewai.cli.run_crew.execute_command")
@mock.patch("crewai.cli.run_crew.read_toml")
@mock.patch("crewai.cli.run_crew.get_crewai_version")
def test_run_flow_with_active_flag(mock_version, mock_toml, mock_execute, runner):
    """Test that run command works with --active flag for flows."""
    mock_version.return_value = "0.148.0"
    mock_toml.return_value = {"tool": {"crewai": {"type": "flow"}}}
    
    result = runner.invoke(run, ["--active"])
    
    assert result.exit_code == 0
    mock_execute.assert_called_once_with(CrewType.FLOW, True)


@mock.patch("crewai.cli.run_crew.subprocess.run")
def test_execute_command_standard_crew_without_active(mock_subprocess_run):
    """Test execute_command for standard crew without active flag."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "run_crew"], returncode=0
    )
    
    execute_command(CrewType.STANDARD, active=False)
    
    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "run_crew"],
        capture_output=False,
        text=True,
        check=True,
    )


@mock.patch("crewai.cli.run_crew.subprocess.run")
def test_execute_command_standard_crew_with_active(mock_subprocess_run):
    """Test execute_command for standard crew with active flag."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "--no-sync", "run_crew"], returncode=0
    )
    
    execute_command(CrewType.STANDARD, active=True)
    
    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "--no-sync", "run_crew"],
        capture_output=False,
        text=True,
        check=True,
    )


@mock.patch("crewai.cli.run_crew.subprocess.run")
def test_execute_command_flow_with_active(mock_subprocess_run):
    """Test execute_command for flow with active flag."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "--no-sync", "kickoff"], returncode=0
    )
    
    execute_command(CrewType.FLOW, active=True)
    
    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "--no-sync", "kickoff"],
        capture_output=False,
        text=True,
        check=True,
    )


@mock.patch("crewai.cli.run_crew.subprocess.run")
def test_execute_command_flow_without_active(mock_subprocess_run):
    """Test execute_command for flow without active flag."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["uv", "run", "kickoff"], returncode=0
    )
    
    execute_command(CrewType.FLOW, active=False)
    
    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "kickoff"],
        capture_output=False,
        text=True,
        check=True,
    )


@mock.patch("crewai.cli.run_crew.subprocess.run")
@mock.patch("crewai.cli.run_crew.click.echo")
def test_execute_command_handles_subprocess_error(mock_echo, mock_subprocess_run):
    """Test that execute_command properly handles subprocess errors."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["uv", "run", "--no-sync", "run_crew"],
        output="Error output"
    )
    
    execute_command(CrewType.STANDARD, active=True)
    
    mock_subprocess_run.assert_called_once_with(
        ["uv", "run", "--no-sync", "run_crew"],
        capture_output=False,
        text=True,
        check=True,
    )


@mock.patch("crewai.cli.run_crew.subprocess.run")
@mock.patch("crewai.cli.run_crew.click.echo")
def test_execute_command_handles_general_exception(mock_echo, mock_subprocess_run):
    """Test that execute_command properly handles general exceptions."""
    mock_subprocess_run.side_effect = Exception("Unexpected error")
    
    execute_command(CrewType.STANDARD, active=True)
    
    mock_echo.assert_called_once_with("An unexpected error occurred: Unexpected error", err=True)
