import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from crewai.cli.run_crew import CrewType, execute_command


@pytest.mark.parametrize("crew_type", [CrewType.STANDARD, CrewType.FLOW])
def test_execute_command_with_different_crew_types(crew_type):
    """
    Test that execute_command works with different crew types.
    
    Verifies that the correct command is executed based on the crew type.
    """
    # Create a temporary directory with a src subdirectory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "src"
        src_path.mkdir()
        
        # Change to the temporary directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Save the original sys.path
            original_sys_path = sys.path.copy()
            
            # Mock subprocess.run to avoid actually running the command
            with patch("subprocess.run") as mock_run:
                # Call execute_command
                execute_command(crew_type)
                
                # Check that the correct command was called based on crew_type
                expected_command = ["uv", "run", "kickoff" if crew_type == CrewType.FLOW else "run_crew"]
                mock_run.assert_called_once_with(expected_command, capture_output=False, text=True, check=True)
        finally:
            # Restore the original directory and sys.path
            os.chdir(original_dir)
            sys.path = original_sys_path


def test_execute_command_handles_missing_src_directory():
    """
    Test that execute_command handles missing src directory gracefully.
    
    Verifies that the command executes even when the src directory doesn't exist.
    """
    # Create a temporary directory without a src subdirectory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Change to the temporary directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Save the original sys.path
            original_sys_path = sys.path.copy()
            
            # Mock subprocess.run to avoid actually running the command
            with patch("subprocess.run") as mock_run:
                # Call execute_command
                execute_command(CrewType.STANDARD)
                
                # Check that sys.path wasn't modified (since src doesn't exist)
                assert sys.path == original_sys_path
                
                # Check that the command was still called
                mock_run.assert_called_once_with(["uv", "run", "run_crew"], capture_output=False, text=True, check=True)
        finally:
            # Restore the original directory and sys.path
            os.chdir(original_dir)
            sys.path = original_sys_path


def test_execute_command_handles_subprocess_error():
    """
    Test that execute_command properly handles subprocess errors.
    
    Verifies that exceptions from subprocess.run are propagated correctly.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        src_path = temp_path / "src"
        src_path.mkdir()
        
        # Create a dummy pyproject.toml file
        with open(temp_path / "pyproject.toml", "w") as f:
            f.write("[project]\nname = \"test\"\n")
        
        # Change to the temporary directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Save the original sys.path
            original_sys_path = sys.path.copy()
            
            # Mock subprocess.run to raise an exception
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, [])):
                # Call execute_command and verify it handles the error
                execute_command(CrewType.STANDARD)
                
                # Verify that sys.path was modified correctly
                assert str(src_path) in sys.path
        finally:
            # Restore the original directory and sys.path
            os.chdir(original_dir)
            sys.path = original_sys_path
