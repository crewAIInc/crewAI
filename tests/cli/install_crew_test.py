import subprocess
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from crewai.cli.install_crew import install_crew


@pytest.fixture
def mock_subprocess_run():
    with mock.patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_path_exists():
    with mock.patch("pathlib.Path.exists") as mock_exists:
        yield mock_exists


def test_install_crew_pyproject_exists(mock_subprocess_run, mock_path_exists):
    mock_path_exists.return_value = True
    proxy_options = []

    install_crew(proxy_options)

    mock_subprocess_run.assert_called_once_with(
        ["uv", "sync"] + proxy_options, check=True, capture_output=False, text=True
    )


def test_install_crew_pyproject_not_exists(mock_subprocess_run, mock_path_exists, capsys):
    mock_path_exists.return_value = False
    proxy_options = []

    install_crew(proxy_options)

    mock_subprocess_run.assert_not_called()
    captured = capsys.readouterr()
    assert "Error: No pyproject.toml found in current directory." in captured.err
    assert "This command must be run from the root of a crew project." in captured.err


def test_install_crew_subprocess_error(mock_subprocess_run, mock_path_exists, capsys):
    mock_path_exists.return_value = True
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        cmd=["uv", "sync"], returncode=1
    )
    proxy_options = []

    install_crew(proxy_options)

    captured = capsys.readouterr()
    assert "An error occurred while running the crew" in captured.err


def test_install_crew_general_exception(mock_subprocess_run, mock_path_exists, capsys):
    mock_path_exists.return_value = True
    mock_subprocess_run.side_effect = Exception("Test exception")
    proxy_options = []

    install_crew(proxy_options)

    captured = capsys.readouterr()
    assert "An unexpected error occurred: Test exception" in captured.err
