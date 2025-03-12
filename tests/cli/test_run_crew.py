import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from crewai.cli.run_crew import execute_command, CrewType


def test_execute_command_adds_src_to_path():
    """Test that execute_command adds the src directory to sys.path."""
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
                execute_command(CrewType.STANDARD)
                
                # Check that src_path was added to sys.path
                assert str(src_path) in sys.path
                
                # Check that the command was called
                mock_run.assert_called_once()
        finally:
            # Restore the original directory and sys.path
            os.chdir(original_dir)
            sys.path = original_sys_path
