import os
import tempfile
from pathlib import Path
import pytest
from crewai.utilities.paths import get_knowledge_directory


class TestKnowledgeDirectory:
    def test_default_knowledge_directory(self):
        """Test that default knowledge directory is returned when env var not set."""
        if "CREWAI_KNOWLEDGE_FILE_DIR" in os.environ:
            del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]
        
        result = get_knowledge_directory()
        assert result == "knowledge"

    def test_custom_knowledge_directory(self):
        """Test that custom directory is returned when env var is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = temp_dir
            try:
                result = get_knowledge_directory()
                assert result == temp_dir
            finally:
                del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]

    def test_invalid_knowledge_directory(self):
        """Test that ValueError is raised for non-existent directory."""
        invalid_dir = "/path/that/does/not/exist"
        os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = invalid_dir
        try:
            with pytest.raises(ValueError, match="Knowledge directory does not exist"):
                get_knowledge_directory()
        finally:
            del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]

    def test_relative_path_knowledge_directory(self):
        """Test that relative paths work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sub_dir = Path(temp_dir) / "knowledge_files"
            sub_dir.mkdir()
            
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = "knowledge_files"
                
                result = get_knowledge_directory()
                assert result == str(sub_dir)
            finally:
                os.chdir(original_cwd)
                if "CREWAI_KNOWLEDGE_FILE_DIR" in os.environ:
                    del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]

    def test_absolute_path_knowledge_directory(self):
        """Test that absolute paths work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = temp_dir
            try:
                result = get_knowledge_directory()
                assert result == temp_dir
            finally:
                del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]
