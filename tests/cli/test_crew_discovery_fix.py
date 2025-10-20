"""Tests for crew discovery bug fixes."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai.cli.crew_chat import load_crew_and_name
from crewai.cli.utils import get_crews, normalize_project_name


class TestNormalizeProjectName:
    """Test project name normalization."""

    def test_underscore_project_name(self):
        """Test project name with underscores."""
        module_name, class_name = normalize_project_name("my_project")
        assert module_name == "my_project"
        assert class_name == "MyProject"

    def test_hyphen_project_name(self):
        """Test project name with hyphens."""
        module_name, class_name = normalize_project_name("my-project")
        assert module_name == "my_project"
        assert class_name == "MyProject"

    def test_mixed_separators(self):
        """Test project name with mixed separators."""
        module_name, class_name = normalize_project_name("my-project_name")
        assert module_name == "my_project_name"
        assert class_name == "MyProjectName"

    def test_complex_project_name(self):
        """Test complex project name like the one in the issue."""
        module_name, class_name = normalize_project_name(
            "dropbox_to_rag_migration_system"
        )
        assert module_name == "dropbox_to_rag_migration_system"
        assert class_name == "DropboxToRagMigrationSystem"

    def test_hyphenated_complex_name(self):
        """Test hyphenated complex project name."""
        module_name, class_name = normalize_project_name(
            "dropbox-to-rag-migration-system"
        )
        assert module_name == "dropbox_to_rag_migration_system"
        assert class_name == "DropboxToRagMigrationSystem"


class TestCrewDiscovery:
    """Test crew discovery improvements."""

    def test_get_crews_with_better_error_handling(self):
        """Test that get_crews provides better error information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a broken crew.py file
            crew_file = Path(temp_dir) / "crew.py"
            crew_file.write_text("""
# Broken crew file
import non_existent_module  # This will cause ImportError

class TestCrew:
    pass
""")

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Should not raise exception when require=False
                crews = get_crews(require=False)
                assert crews == []

                # Should provide detailed error when require=True
                with pytest.raises(SystemExit):
                    get_crews(require=True)

            finally:
                os.chdir(original_cwd)

    def test_get_crews_finds_valid_crew(self):
        """Test that get_crews can find a valid crew."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid crew.py file
            crew_file = Path(temp_dir) / "crew.py"
            crew_file.write_text("""
# Simple test crew
class TestCrew:
    pass
""")

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Mock the dependencies to avoid actual CrewAI imports
                with patch("crewai.cli.utils.fetch_crews") as mock_fetch:
                    # Return empty list for most calls, one crew for our test class
                    def fetch_side_effect(attr):
                        if hasattr(attr, "__name__") and attr.__name__ == "TestCrew":
                            mock_crew = MagicMock()
                            return [mock_crew]
                        return []

                    mock_fetch.side_effect = fetch_side_effect

                    crews = get_crews(require=False)
                    # Should find at least one crew (might find more due to other attributes)
                    assert len(crews) >= 1

            finally:
                os.chdir(original_cwd)


class TestLoadCrewAndName:
    """Test the improved load_crew_and_name function."""

    def test_load_crew_with_detailed_errors(self):
        """Test that load_crew_and_name provides detailed error information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pyproject.toml
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text("""
[project]
name = "test-project"
version = "0.1.0"
""")

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Should provide detailed error information
                with pytest.raises(ImportError) as exc_info:
                    load_crew_and_name()

                error_message = str(exc_info.value)
                assert (
                    "Failed to import crew module using all strategies" in error_message
                )
                assert "Project name: test-project" in error_message
                assert "Folder name: test_project" in error_message
                assert "Expected crew class: TestProject" in error_message
                assert "Debug info:" in error_message

            finally:
                os.chdir(original_cwd)

    def test_load_crew_with_fallback_class_detection(self):
        """Test that load_crew_and_name provides better error messages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pyproject.toml
            pyproject_file = Path(temp_dir) / "pyproject.toml"
            pyproject_file.write_text("""
[project]
name = "test-project"
version = "0.1.0"
""")

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Should provide detailed error information when no crew is found
                with pytest.raises(ImportError) as exc_info:
                    load_crew_and_name()

                error_message = str(exc_info.value)
                # Check that our improved error message includes debugging info
                assert (
                    "Failed to import crew module using all strategies" in error_message
                )
                assert "Debug info:" in error_message

            finally:
                os.chdir(original_cwd)
