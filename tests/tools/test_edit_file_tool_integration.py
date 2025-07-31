import tempfile
import os
from unittest.mock import patch
from crewai import Agent
from crewai.tools.agent_tools.edit_file_tool import EditFileTool


class TestEditFileToolIntegration:
    
    def test_agent_with_edit_file_tool(self):
        """Test that agents can use the EditFileTool."""
        tool = EditFileTool()
        
        agent = Agent(
            role="Code Editor",
            goal="Edit files as requested",
            backstory="I am an expert at editing code files",
            tools=[tool],
            verbose=True
        )
        
        assert tool in agent.tools
        assert agent.tools[0].name == "edit_file"
        
    def test_tool_with_different_file_types(self):
        """Test the tool works with different file types."""
        tool = EditFileTool()
        
        test_files = [
            ("test.py", "print('hello')", "Change hello to world"),
            ("test.js", "console.log('hello');", "Change hello to world"),
            ("test.txt", "Hello world", "Change Hello to Hi"),
            ("test.md", "# Title\nContent", "Change Title to New Title")
        ]
        
        for filename, content, instruction in test_files:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=filename) as temp_file:
                temp_file.write(content)
                temp_file.flush()
                
                try:
                    with patch.object(tool.llm, 'call') as mock_call:
                        expected_content = content.replace('hello', 'world').replace('Hello', 'Hi').replace('Title', 'New Title')
                        mock_call.return_value = expected_content
                        
                        result = tool._run(
                            file_path=temp_file.name,
                            edit_instructions=instruction
                        )
                        
                        assert "Successfully edited" in result
                        
                        with open(temp_file.name, 'r') as f:
                            edited_content = f.read()
                        assert edited_content == expected_content
                        
                        backup_path = f"{temp_file.name}.backup"
                        if os.path.exists(backup_path):
                            os.unlink(backup_path)
                            
                finally:
                    os.unlink(temp_file.name)
