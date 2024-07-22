import subprocess
from unittest import mock

import pytest

from crewai.cli import test_crew


@pytest.mark.parametrize(
    "n_iterations,model",
    [
        (1, "gpt-4o"),
        (5, "gpt-3.5-turbo"),
        (10, "gpt-4"),
    ],
)
@mock.patch("crewai.cli.test_crew.subprocess.run")
def test_crew_success(mock_subprocess_run, n_iterations, model):
    """Test the crew function for successful execution."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=f"poetry run test {n_iterations} {model}", returncode=0
    )
    result = test_crew.test_crew(n_iterations, model)

    mock_subprocess_run.assert_called_once_with(
        ["poetry", "run", "test", str(n_iterations), model],
        capture_output=False,
        text=True,
        check=True,
    )
    assert result is None


@mock.patch("crewai.cli.test_crew.click")
def test_test_crew_zero_iterations(click):
    test_crew.test_crew(0, "gpt-4o")
    click.echo.assert_called_once_with(
        "An unexpected error occurred: The number of iterations must be a positive integer.",
        err=True,
    )


@mock.patch("crewai.cli.test_crew.click")
def test_test_crew_negative_iterations(click):
    test_crew.test_crew(-2, "gpt-4o")
    click.echo.assert_called_once_with(
        "An unexpected error occurred: The number of iterations must be a positive integer.",
        err=True,
    )
