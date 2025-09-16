import os
import tempfile
from pathlib import Path
import pytest
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource


class TestExternalKnowledgeDirectory:
    def test_text_file_source_with_external_directory(self):
        """Test that TextFileKnowledgeSource works with external directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_content = "This is a test file for external knowledge directory."
            test_file.write_text(test_content)
            
            os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = temp_dir
            try:
                source = TextFileKnowledgeSource(file_paths=["test.txt"])
                
                assert len(source.content) == 1
                loaded_content = list(source.content.values())[0]
                assert loaded_content == test_content
                
            finally:
                del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]

    def test_json_file_source_with_external_directory(self):
        """Test that JSONKnowledgeSource works with external directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"
            test_data = {"name": "John", "age": 30, "city": "New York"}
            import json
            test_file.write_text(json.dumps(test_data))
            
            os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = temp_dir
            try:
                source = JSONKnowledgeSource(file_paths=["test.json"])
                
                assert len(source.content) == 1
                loaded_content = list(source.content.values())[0]
                assert "John" in loaded_content
                assert "30" in loaded_content
                assert "New York" in loaded_content
                
            finally:
                del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]

    def test_knowledge_source_fallback_to_default(self):
        """Test that knowledge sources fall back to default directory when env var not set."""
        if "CREWAI_KNOWLEDGE_FILE_DIR" in os.environ:
            del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]
        
        knowledge_dir = Path("knowledge")
        knowledge_dir.mkdir(exist_ok=True)
        test_file = knowledge_dir / "test_fallback.txt"
        test_content = "This is a test file for default knowledge directory."
        
        try:
            test_file.write_text(test_content)
            
            source = TextFileKnowledgeSource(file_paths=["test_fallback.txt"])
            
            assert len(source.content) == 1
            loaded_content = list(source.content.values())[0]
            assert loaded_content == test_content
            
        finally:
            if test_file.exists():
                test_file.unlink()

    def test_knowledge_source_with_absolute_path_ignores_env_var(self):
        """Test that absolute paths ignore the environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_absolute.txt"
            test_content = "This is a test file with absolute path."
            test_file.write_text(test_content)
            
            with tempfile.TemporaryDirectory() as other_dir:
                os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = other_dir
                try:
                    source = TextFileKnowledgeSource(file_paths=[str(test_file)])
                    
                    assert len(source.content) == 1
                    loaded_content = list(source.content.values())[0]
                    assert loaded_content == test_content
                    
                finally:
                    del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]

    def test_knowledge_source_error_with_invalid_external_directory(self):
        """Test that proper error is raised when external directory doesn't exist."""
        invalid_dir = "/path/that/does/not/exist"
        os.environ["CREWAI_KNOWLEDGE_FILE_DIR"] = invalid_dir
        try:
            with pytest.raises(ValueError, match="Knowledge directory does not exist"):
                TextFileKnowledgeSource(file_paths=["test.txt"])
        finally:
            del os.environ["CREWAI_KNOWLEDGE_FILE_DIR"]
