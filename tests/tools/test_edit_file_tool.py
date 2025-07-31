import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from crewai.tools.agent_tools.edit_file_tool import EditFileTool, EditFileToolInput


class TestEditFileTool:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = EditFileTool()
        
    def test_tool_initialization(self):
        """Test that the tool initializes correctly."""
        assert self.tool.name == "edit_file"
        assert "Fast Apply" in self.tool.description
        assert self.tool.args_schema == EditFileToolInput
        
    def test_input_schema_validation(self):
        """Test input schema validation."""
        valid_input = EditFileToolInput(
            file_path="/path/to/file.py",
            edit_instructions="Add a new function",
            context="This is for testing"
        )
        assert valid_input.file_path == "/path/to/file.py"
        assert valid_input.edit_instructions == "Add a new function"
        assert valid_input.context == "This is for testing"
        
        with pytest.raises(ValueError):
            EditFileToolInput(file_path="/path/to/file.py")
            
    def test_file_not_exists(self):
        """Test handling of non-existent files."""
        result = self.tool._run(
            file_path="/non/existent/file.py",
            edit_instructions="Add a function"
        )
        assert "Error: File /non/existent/file.py does not exist" in result
        
    def test_path_is_directory(self):
        """Test handling when path points to a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.tool._run(
                file_path=temp_dir,
                edit_instructions="Add a function"
            )
            assert f"Error: {temp_dir} is not a file" in result
            
    def test_binary_file_handling(self):
        """Test handling of binary files."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as temp_file:
            temp_file.write(b'\x00\x01\x02\x03')
            temp_file.flush()
            
            try:
                result = self.tool._run(
                    file_path=temp_file.name,
                    edit_instructions="Edit this file"
                )
                assert "Cannot read" in result and "unsupported encoding" in result
            finally:
                os.unlink(temp_file.name)
                
    @patch('crewai.tools.agent_tools.edit_file_tool.LLM')
    def test_successful_file_edit(self, mock_llm_class):
        """Test successful file editing."""
        mock_llm = Mock()
        mock_llm.call.return_value = "def new_function():\n    return 'edited'"
        mock_llm_class.return_value = mock_llm
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
            temp_file.write("def old_function():\n    return 'original'")
            temp_file.flush()
            
            try:
                tool = EditFileTool()
                result = tool._run(
                    file_path=temp_file.name,
                    edit_instructions="Replace old_function with new_function"
                )
                
                assert "Successfully edited" in result
                assert "Backup saved" in result
                
                with open(temp_file.name, 'r') as f:
                    content = f.read()
                assert "new_function" in content
                assert "old_function" not in content
                
                backup_path = f"{temp_file.name}.backup"
                assert os.path.exists(backup_path)
                with open(backup_path, 'r') as f:
                    backup_content = f.read()
                assert "old_function" in backup_content
                
                os.unlink(backup_path)
                
            finally:
                os.unlink(temp_file.name)
                
    def test_build_fast_apply_prompt(self):
        """Test Fast Apply prompt building."""
        prompt = self.tool._build_fast_apply_prompt(
            current_content="print('hello')",
            edit_instructions="Change hello to world",
            file_path="/test/file.py",
            context="Testing context"
        )
        
        assert "Fast Apply file editing" in prompt
        assert "print('hello')" in prompt
        assert "Change hello to world" in prompt
        assert "/test/file.py" in prompt
        assert "Testing context" in prompt
        assert "COMPLETE rewritten file content" in prompt
        
    def test_extract_file_content_plain(self):
        """Test extracting content from plain LLM response."""
        response = "def hello():\n    print('world')"
        content = self.tool._extract_file_content(response)
        assert content == "def hello():\n    print('world')"
        
    def test_extract_file_content_markdown(self):
        """Test extracting content from markdown code blocks."""
        response = "```python\ndef hello():\n    print('world')\n```"
        content = self.tool._extract_file_content(response)
        assert content == "def hello():\n    print('world')"
        
    def test_extract_file_content_empty(self):
        """Test handling empty LLM response."""
        content = self.tool._extract_file_content("")
        assert content is None
        
        content = self.tool._extract_file_content("```\n```")
        assert content == ""
        
    @patch('crewai.tools.agent_tools.edit_file_tool.LLM')
    def test_llm_failure_handling(self, mock_llm_class):
        """Test handling of LLM failures."""
        mock_llm = Mock()
        mock_llm.call.side_effect = Exception("LLM error")
        mock_llm_class.return_value = mock_llm
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as temp_file:
            temp_file.write("original content")
            temp_file.flush()
            
            try:
                tool = EditFileTool()
                result = tool._run(
                    file_path=temp_file.name,
                    edit_instructions="Edit this file"
                )
                
                assert "Error editing file" in result
                assert "LLM error" in result
                
            finally:
                os.unlink(temp_file.name)
                
    def test_context_parameter(self):
        """Test that context parameter is properly handled."""
        prompt_with_context = self.tool._build_fast_apply_prompt(
            current_content="test",
            edit_instructions="edit",
            file_path="/test.py",
            context="important context"
        )
        
        prompt_without_context = self.tool._build_fast_apply_prompt(
            current_content="test", 
            edit_instructions="edit",
            file_path="/test.py",
            context=None
        )
        
        assert "important context" in prompt_with_context
        assert "ADDITIONAL CONTEXT" in prompt_with_context
        assert "ADDITIONAL CONTEXT" not in prompt_without_context
