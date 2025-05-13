import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from crewai.tools.tool_usage import ToolUsage


class TestComplexInput(BaseModel):
    special_chars: str = Field(
        ..., description="Parameter with special characters: \"'\\{}[]"
    )
    nested_dict: dict = Field(
        ..., description="A nested dictionary parameter"
    )
    unicode_text: str = Field(
        ..., description="Text with unicode characters: 你好, こんにちは, مرحبا"
    )


class TestComplexTool(BaseTool):
    name: str = "Complex JSON Tool"
    description: str = "A tool for testing complex JSON formatting"
    args_schema: type[BaseModel] = TestComplexInput

    def _run(self, special_chars: str, nested_dict: dict, unicode_text: str) -> str:
        return f"Processed complex input successfully"


def test_complex_json_formatting():
    """Test that complex JSON with special characters and nested structures is formatted correctly."""
    tool = TestComplexTool()
    
    assert "Tool Arguments:" in tool.description
    
    description_parts = tool.description.split("Tool Arguments: ")
    json_str = description_parts[1].split("\nTool Description:")[0]
    
    parsed_json = json.loads(json_str)
    
    assert "special_chars" in parsed_json
    assert "nested_dict" in parsed_json
    assert "unicode_text" in parsed_json
    
    assert "\"'\\{}[]" in parsed_json["special_chars"]["description"]
    
    assert "你好" in parsed_json["unicode_text"]["description"]
    assert "こんにちは" in parsed_json["unicode_text"]["description"]
    assert "مرحبا" in parsed_json["unicode_text"]["description"]


def test_complex_tool_usage_render():
    """Test that complex tool usage renders with proper JSON formatting."""
    tool = TestComplexTool()

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[tool],
        original_tools=[tool],
        tools_description="Tool for testing complex JSON formatting",
        tools_names="test_complex_tool",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    rendered = tool_usage._render()
    
    rendered_parts = rendered.split("Tool Arguments: ")
    if len(rendered_parts) > 1:
        json_str = rendered_parts[1].split("\nTool Description:")[0]
        
        try:
            parsed_json = json.loads(json_str)
            assert True  # If we get here, JSON parsing succeeded
            
            assert "special_chars" in parsed_json
            assert "nested_dict" in parsed_json
            assert "unicode_text" in parsed_json
            
        except json.JSONDecodeError:
            assert False, "The rendered tool arguments are not valid JSON"
