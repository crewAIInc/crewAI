from typing import Type

import pytest
from pydantic import BaseModel, Field

from crewai.tools import BaseTool


class TestToolInput(BaseModel):
    param: str = Field(description="A test parameter")


class TestTool(BaseTool):
    name: str = "Test Tool"
    description: str = "A tool for testing the invoke method"
    args_schema: Type[BaseModel] = TestToolInput

    def _run(self, param: str) -> str:
        return f"Tool executed with: {param}"


def test_invoke_with_invalid_type():
    """Test that invoke raises ValueError with invalid input types."""
    tool = TestTool()
    with pytest.raises(ValueError, match="Input must be string or dict"):
        tool.invoke(input=123)
    
    with pytest.raises(ValueError, match="Input must be string or dict"):
        tool.invoke(input=["list", "not", "allowed"])
    
    with pytest.raises(ValueError, match="Input must be string or dict"):
        tool.invoke(input=None)


def test_invoke_with_config():
    """Test that invoke properly handles configuration dictionaries."""
    tool = TestTool()
    # Config should be passed through to _run but not affect the result
    result = tool.invoke(input={"param": "test with config"}, config={"timeout": 30})
    assert result == "Tool executed with: test with config"


def test_invoke_with_malformed_json():
    """Test that invoke handles malformed JSON gracefully."""
    tool = TestTool()
    # Malformed JSON should be treated as a raw string
    result = tool.invoke(input="{param: this is not valid JSON}")
    assert "this is not valid JSON" in result


def test_invoke_with_nested_dict():
    """Test that invoke handles nested dictionaries properly."""
    class NestedToolInput(BaseModel):
        config: dict = Field(description="A nested configuration dictionary")
    
    class NestedTool(BaseTool):
        name: str = "Nested Tool"
        description: str = "A tool for testing nested dictionaries"
        args_schema: Type[BaseModel] = NestedToolInput
        
        def _run(self, config: dict) -> str:
            return f"Tool executed with nested config: {config}"
    
    tool = NestedTool()
    nested_input = {"config": {"key1": "value1", "key2": {"nested": "value"}}}
    result = tool.invoke(input=nested_input)
    assert "Tool executed with nested config" in result
    assert "key1" in result
    assert "nested" in result
