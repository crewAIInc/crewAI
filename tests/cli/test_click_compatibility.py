import subprocess
import sys
import tempfile

import pytest


def test_crewai_cli_works_with_compatible_click_version():
    """Test that crewAI CLI works with Click version compatible with zenml."""
    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a new virtual environment
        subprocess.run(
            [sys.executable, "-m", "venv", f"{temp_dir}/venv"],
            check=True,
        )
        
        # Install Click 8.1.3 (compatible with zenml's constraints)
        subprocess.run(
            [f"{temp_dir}/venv/bin/pip", "install", "click==8.1.3"],
            check=True,
        )
        
        # Install crewai in development mode
        subprocess.run(
            [f"{temp_dir}/venv/bin/pip", "install", "-e", "."],
            check=True,
        )
        
        # Verify that the crewai CLI can be imported and run
        result = subprocess.run(
            [f"{temp_dir}/venv/bin/python", "-c", "from crewai.cli.cli import crewai; print('CLI import successful')"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "CLI import successful" in result.stdout
        
        # Test running a basic CLI command
        result = subprocess.run(
            [f"{temp_dir}/venv/bin/python", "-m", "crewai.cli.cli", "version"],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0
        assert "crewai version" in result.stdout
