import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from crewai.cli.cli import reset_memories
from crewai.cli.reset_memories_command import reset_memories_command


def test_reset_memories_command_parameters():
    """Test that the CLI parameters match the function parameters."""
    # Create a mock for reset_memories_command
    with patch('crewai.cli.cli.reset_memories_command') as mock_reset:
        runner = CliRunner()
        
        # Test with entities flag
        result = runner.invoke(reset_memories, ['--entities'])
        assert result.exit_code == 0
        
        # Check that the function was called with the correct parameters
        # The third parameter should be True for entities
        mock_reset.assert_called_once_with(False, False, True, False, False, False)


def test_reset_memories_all_flag():
    """Test that the --all flag resets all memories."""
    with patch('crewai.cli.cli.reset_memories_command') as mock_reset:
        runner = CliRunner()
        
        # Test with all flag
        result = runner.invoke(reset_memories, ['--all'])
        assert result.exit_code == 0
        
        # Check that the function was called with the correct parameters
        # The last parameter should be True for all
        mock_reset.assert_called_once_with(False, False, False, False, False, True)


def test_reset_memories_knowledge_flag():
    """Test that the --knowledge flag resets knowledge storage."""
    with patch('crewai.cli.cli.reset_memories_command') as mock_reset:
        runner = CliRunner()
        
        # Test with knowledge flag
        result = runner.invoke(reset_memories, ['--knowledge'])
        assert result.exit_code == 0
        
        # Check that the function was called with the correct parameters
        # The fourth parameter should be True for knowledge
        mock_reset.assert_called_once_with(False, False, False, True, False, False)


def test_reset_memories_no_flags():
    """Test that an error message is shown when no flags are provided."""
    runner = CliRunner()
    
    # Test with no flags
    result = runner.invoke(reset_memories, [])
    assert result.exit_code == 0
    assert "Please specify at least one memory type" in result.output
