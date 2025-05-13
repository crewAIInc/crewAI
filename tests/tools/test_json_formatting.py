import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from crewai.tools.tool_usage import ToolUsage


class TestJsonInput(BaseModel):
    test_param: str = Field(
        ..., description="A test parameter"
    )
    another_param: int = Field(
        ..., description="Another test parameter"
    )


class TestJsonTool(BaseTool):
    name: str = "Test JSON Tool"
    description: str = "A tool for testing JSON formatting"
    args_schema: type[BaseModel] = TestJsonInput

    def _run(self, test_param: str, another_param: int) -> str:
        return f"Received {test_param} and {another_param}"


def test_tool_description_json_formatting():
    """Test that the tool description uses proper JSON formatting with double quotes."""
    tool = TestJsonTool()
    
    assert "Tool Arguments:" in tool.description
    
    description_parts = tool.description.split("Tool Arguments: ")
    json_str = description_parts[1].split("\nTool Description:")[0]
    
    parsed_json = json.loads(json_str)
    
    assert "test_param" in parsed_json
    assert "another_param" in parsed_json
    
    assert '"test_param"' in json_str
    assert '"another_param"' in json_str
    assert "'" not in json_str  # No single quotes should be present


def test_tool_usage_json_formatting():
    """Test that the tool usage renders with proper JSON formatting."""
    tool = TestJsonTool()

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[tool],
        original_tools=[tool],
        tools_description="Tool for testing JSON formatting",
        tools_names="test_json_tool",
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
        except json.JSONDecodeError:
            assert False, "The rendered tool arguments are not valid JSON"
        
        assert '"test_param"' in json_str
        assert '"another_param"' in json_str
        assert "'" not in json_str  # No single quotes should be present
