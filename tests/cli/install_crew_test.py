import subprocess
from pathlib import Path
from typing import List
from unittest import mock

import pytest
from click.testing import CliRunner

from crewai.cli.install_crew import install_crew, _check_pyproject_exists

TEST_CONSTANTS = {
    "ERROR_NO_PYPROJECT": "Error: No pyproject.toml found in current directory.",
    "ERROR_MUST_RUN_FROM_ROOT": "This command must be run from the root of a crew project.",
    "ERROR_RUNNING_CREW": "An error occurred while running the crew",
    "ERROR_UNEXPECTED": "An unexpected error occurred: Test exception",
}


@pytest.fixture
def mock_subprocess_run() -> mock.MagicMock:
    """Mock subprocess.run for testing."""
    with mock.patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_path_exists() -> mock.MagicMock:
    """Mock Path.exists for testing."""
    with mock.patch("pathlib.Path.exists") as mock_exists:
        yield mock_exists


@pytest.mark.parametrize(
    "proxy_options,expected_command",
    [
        ([], ["uv", "sync"]),
        (["--proxy", "http://proxy.com"], ["uv", "sync", "--proxy", "http://proxy.com"]),
    ],
)
def test_install_crew_pyproject_exists(
    mock_subprocess_run: mock.MagicMock,
    mock_path_exists: mock.MagicMock,
    proxy_options: List[str],
    expected_command: List[str],
) -> None:
    """Test install_crew when pyproject.toml exists."""
    mock_path_exists.return_value = True

    install_crew(proxy_options)

    mock_subprocess_run.assert_called_once_with(
        expected_command, check=True, capture_output=False, text=True
    )


def test_install_crew_pyproject_not_exists(
    mock_subprocess_run: mock.MagicMock,
    mock_path_exists: mock.MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test install_crew when pyproject.toml does not exist."""
    mock_path_exists.return_value = False
    proxy_options: List[str] = []

    install_crew(proxy_options)

    mock_subprocess_run.assert_not_called()
    captured = capsys.readouterr()
    assert TEST_CONSTANTS["ERROR_NO_PYPROJECT"] in captured.err
    assert TEST_CONSTANTS["ERROR_MUST_RUN_FROM_ROOT"] in captured.err


def test_install_crew_subprocess_error(
    mock_subprocess_run: mock.MagicMock,
    mock_path_exists: mock.MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test install_crew when subprocess raises CalledProcessError."""
    mock_path_exists.return_value = True
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        cmd=["uv", "sync"], returncode=1
    )
    proxy_options: List[str] = []

    install_crew(proxy_options)

    captured = capsys.readouterr()
    assert TEST_CONSTANTS["ERROR_RUNNING_CREW"] in captured.err


def test_install_crew_general_exception(
    mock_subprocess_run: mock.MagicMock,
    mock_path_exists: mock.MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test install_crew when a general exception occurs."""
    mock_path_exists.return_value = True
    mock_subprocess_run.side_effect = Exception("Test exception")
    proxy_options: List[str] = []

    install_crew(proxy_options)

    captured = capsys.readouterr()
    assert TEST_CONSTANTS["ERROR_UNEXPECTED"] in captured.err


def test_check_pyproject_exists(
    mock_path_exists: mock.MagicMock, capsys: pytest.CaptureFixture
) -> None:
    """Test _check_pyproject_exists function."""
    mock_path_exists.return_value = True
    assert _check_pyproject_exists() is True

    mock_path_exists.return_value = False
    assert _check_pyproject_exists() is False
    captured = capsys.readouterr()
    assert TEST_CONSTANTS["ERROR_NO_PYPROJECT"] in captured.err
    assert TEST_CONSTANTS["ERROR_MUST_RUN_FROM_ROOT"] in captured.err
