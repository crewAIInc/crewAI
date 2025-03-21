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


def test_invoke_with_dict():
    """Test that invoke works with a dictionary input."""
    tool = TestTool()
    result = tool.invoke(input={"param": "test value"})
    assert result == "Tool executed with: test value"


def test_invoke_with_json_string():
    """Test that invoke works with a JSON string input."""
    tool = TestTool()
    result = tool.invoke(input='{"param": "test value"}')
    assert result == "Tool executed with: test value"


def test_invoke_with_raw_string():
    """Test that invoke works with a raw string input."""
    tool = TestTool()
    result = tool.invoke(input="test value")
    assert result == "Tool executed with: test value"


def test_invoke_with_empty_dict():
    """Test that invoke handles empty dict input appropriately."""
    tool = TestTool()
    with pytest.raises(Exception):
        # Should raise an exception since param is required
        tool.invoke(input={})


def test_invoke_with_extra_args():
    """Test that invoke filters out extra arguments not in the schema."""
    tool = TestTool()
    result = tool.invoke(input={"param": "test value", "extra": "ignored"})
    assert result == "Tool executed with: test value"
