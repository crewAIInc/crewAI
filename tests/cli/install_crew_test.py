from unittest import mock
import pytest
import subprocess

from crewai.cli.install_crew import install_crew


@mock.patch("subprocess.run")
def test_install_crew_with_active_flag(mock_subprocess):
    """Test that install_crew includes the --active flag."""
    install_crew([])
    mock_subprocess.assert_called_once_with(
        ["uv", "sync", "--active"], check=True, capture_output=False, text=True
    )


@mock.patch("subprocess.run")
def test_install_crew_with_proxy_options(mock_subprocess):
    """Test that install_crew correctly passes proxy options."""
    proxy_options = ["--index-url", "https://custom-pypi.org/simple"]
    install_crew(proxy_options)
    mock_subprocess.assert_called_once_with(
        ["uv", "sync", "--active", "--index-url", "https://custom-pypi.org/simple"],
        check=True,
        capture_output=False,
        text=True,
    )


@mock.patch("subprocess.run")
@mock.patch("click.echo")
def test_install_crew_with_subprocess_error(mock_echo, mock_subprocess):
    """Test that install_crew handles subprocess errors correctly."""
    error = subprocess.CalledProcessError(1, "uv sync --active")
    error.output = "Error output"
    mock_subprocess.side_effect = error
    
    install_crew([])
    
    assert mock_echo.call_count == 2
    mock_echo.assert_any_call(f"An error occurred while running the crew: {error}", err=True)
    mock_echo.assert_any_call("Error output", err=True)


@mock.patch("subprocess.run")
@mock.patch("click.echo")
def test_install_crew_with_generic_exception(mock_echo, mock_subprocess):
    """Test that install_crew handles generic exceptions correctly."""
    error = Exception("Generic error")
    mock_subprocess.side_effect = error
    
    install_crew([])
    
    mock_echo.assert_called_once_with(f"An unexpected error occurred: {error}", err=True)
