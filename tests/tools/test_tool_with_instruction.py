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
