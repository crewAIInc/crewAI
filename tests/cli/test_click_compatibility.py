import subprocess
import sys
import tempfile

import pytest

# Constants for test configuration
VENV_PATH = "venv"
TEST_COMMAND = "version"
EXPECTED_OUTPUT = "crewai version"
IMPORT_SUCCESS_MESSAGE = "CLI import successful"


def run_subprocess(*args, **kwargs):
    """Helper function to run subprocess with error handling."""
    try:
        result = subprocess.run(*args, **kwargs)
        result.check_returncode()
        return result
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Subprocess failed with exit code {e.returncode}: {e.stderr if hasattr(e, 'stderr') else ''}")


@pytest.mark.parametrize("click_version", ["8.0.1", "8.1.3", "8.1.4"])
def test_crewai_cli_works_with_compatible_click_version(click_version):
    """
    Verifies crewAI CLI works with multiple Click versions.
    
    Dependencies:
        - Click versions compatible with zenml constraints
    
    Parameters:
        click_version: The version of Click to test with
    """
    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = f"{temp_dir}/{VENV_PATH}"
        
        # Create a new virtual environment
        run_subprocess(
            [sys.executable, "-m", "venv", venv_path],
            check=True,
        )
        
        # Install specific Click version (compatible with zenml's constraints)
        run_subprocess(
            [f"{venv_path}/bin/pip", "install", f"click=={click_version}"],
            check=True,
        )
        
        # Install crewai in development mode
        run_subprocess(
            [f"{venv_path}/bin/pip", "install", "-e", "."],
            check=True,
        )
        
        # Verify that the crewai CLI can be imported and run
        result = run_subprocess(
            [f"{venv_path}/bin/python", "-c", "from crewai.cli.cli import crewai; print('CLI import successful')"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"CLI import failed with output: {result.stderr}"
        assert IMPORT_SUCCESS_MESSAGE in result.stdout, "Failed to import CLI module"
        
        # Test running a basic CLI command
        result = run_subprocess(
            [f"{venv_path}/bin/python", "-m", "crewai.cli.cli", TEST_COMMAND],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"CLI command failed with output: {result.stderr}"
        assert EXPECTED_OUTPUT in result.stdout, f"Expected output '{EXPECTED_OUTPUT}' not found in command result"
