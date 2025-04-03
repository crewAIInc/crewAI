import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, Optional

from crewai.tools.base_tool import BaseTool, Tool
from crewai.tools.tool_with_instruction import ToolWithInstruction


class MockTool(BaseTool):
    """Mock tool for testing."""
    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    
    def _run(self, *args: Any, **kwargs: Any) -> str:
        return "mock result"


class TestToolWithInstruction:
    """Test suite for ToolWithInstruction."""
    
    def test_initialization(self):
        """Test tool initialization with instructions."""
        tool = MockTool()
        instructions = "Only use this tool for XYZ"
        
        wrapped_tool = ToolWithInstruction(tool=tool, instructions=instructions)
        
        assert wrapped_tool.name == tool.name
        assert "Instructions: Only use this tool for XYZ" in wrapped_tool.description
        assert wrapped_tool.args_schema == tool.args_schema
    
    def test_run_method(self):
        """Test that the run method delegates to the original tool."""
        tool = MockTool()
        instructions = "Only use this tool for XYZ"
        
        wrapped_tool = ToolWithInstruction(tool=tool, instructions=instructions)
        result = wrapped_tool.run()
        
        assert result == "mock result"
    
    def test_to_structured_tool(self):
        """Test that to_structured_tool includes instructions."""
        tool = MockTool()
        instructions = "Only use this tool for XYZ"
        
        wrapped_tool = ToolWithInstruction(tool=tool, instructions=instructions)
        structured_tool = wrapped_tool.to_structured_tool()
        
        assert "Instructions: Only use this tool for XYZ" in structured_tool.description
    
    def test_with_function_tool(self):
        """Test tool wrapping with a function tool."""
        def sample_func():
            return "sample result"
            
        tool = Tool(
            name="sample_tool", 
            description="A sample tool", 
            func=sample_func
        )
        
        instructions = "Only use this tool for XYZ"
        wrapped_tool = ToolWithInstruction(tool=tool, instructions=instructions)
        
        assert wrapped_tool.name == tool.name
        assert "Instructions: Only use this tool for XYZ" in wrapped_tool.description
    
    def test_empty_instructions(self):
        """Test that empty instructions raise ValueError."""
        tool = MockTool()
        
        with pytest.raises(ValueError, match="Instructions cannot be empty"):
            ToolWithInstruction(tool=tool, instructions="")
        
        with pytest.raises(ValueError, match="Instructions cannot be empty"):
            ToolWithInstruction(tool=tool, instructions="   ")
    
    def test_too_long_instructions(self):
        """Test that instructions exceeding maximum length raise ValueError."""
        tool = MockTool()
        long_instructions = "x" * (ToolWithInstruction.MAX_INSTRUCTION_LENGTH + 1)
        
        with pytest.raises(ValueError, match="Instructions exceed maximum length"):
            ToolWithInstruction(tool=tool, instructions=long_instructions)
    
    def test_update_instructions(self):
        """Test updating instructions dynamically."""
        tool = MockTool()
        initial_instructions = "Initial instructions"
        new_instructions = "Updated instructions"
        
        wrapped_tool = ToolWithInstruction(tool=tool, instructions=initial_instructions)
        assert "Instructions: Initial instructions" in wrapped_tool.description
        
        wrapped_tool.update_instructions(new_instructions)
        assert "Instructions: Updated instructions" in wrapped_tool.description
        assert wrapped_tool.instructions == new_instructions
    
    def test_update_instructions_validation(self):
        """Test validation when updating instructions."""
        tool = MockTool()
        wrapped_tool = ToolWithInstruction(tool=tool, instructions="Valid instructions")
        
        with pytest.raises(ValueError, match="Instructions cannot be empty"):
            wrapped_tool.update_instructions("")
        
        long_instructions = "x" * (ToolWithInstruction.MAX_INSTRUCTION_LENGTH + 1)
        with pytest.raises(ValueError, match="Instructions exceed maximum length"):
            wrapped_tool.update_instructions(long_instructions)
