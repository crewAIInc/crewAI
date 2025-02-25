import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from crewai.cli.create_crew import create_crew


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_project_formatting(temp_dir):
    """Test that created projects follow PEP8 conventions."""
    # Change to the temporary directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)

    try:
        # Create a new crew project
        create_crew("test_crew", skip_provider=True)

        # Create a ruff configuration file
        ruff_config = """
line-length = 120
target-version = "py310"
select = ["E", "F", "I", "UP", "A"]
ignore = ["D203"]
"""
        with open(Path(temp_dir) / "test_crew" / ".ruff.toml", "w") as f:
            f.write(ruff_config)

        # Run ruff on the generated project code
        result = subprocess.run(
            ["ruff", "check", "test_crew"],
            capture_output=True,
            text=True,
        )

        # Check that there are no linting errors
        assert result.returncode == 0, f"Ruff found issues: {result.stdout}"
        # If ruff reports "All checks passed!" or empty output, that's good
        assert "All checks passed!" in result.stdout or not result.stdout.strip(), f"Ruff found issues: {result.stdout}"

    finally:
        # Change back to the original directory
        os.chdir(original_dir)
