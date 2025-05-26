import pytest
from unittest.mock import MagicMock, patch

from crewai.tools import BaseTool, tool


def test_tool_usage_limit():
    """Test that tools respect usage limits."""
    class LimitedTool(BaseTool):
        name: str = "Limited Tool"
        description: str = "A tool with usage limits for testing"
        max_usage_count: int = 2

        def _run(self, input_text: str) -> str:
            return f"Processed {input_text}"

    tool = LimitedTool()
    
    result1 = tool.run(input_text="test1")
    assert result1 == "Processed test1"
    assert tool.current_usage_count == 1
    
    result2 = tool.run(input_text="test2")
    assert result2 == "Processed test2"
    assert tool.current_usage_count == 2


def test_unlimited_tool_usage():
    """Test that tools without usage limits work normally."""
    class UnlimitedTool(BaseTool):
        name: str = "Unlimited Tool"
        description: str = "A tool without usage limits"

        def _run(self, input_text: str) -> str:
            return f"Processed {input_text}"

    tool = UnlimitedTool()
    
    for i in range(5):
        result = tool.run(input_text=f"test{i}")
        assert result == f"Processed test{i}"
        assert tool.current_usage_count == i + 1


def test_tool_decorator_with_usage_limit():
    """Test usage limit with @tool decorator."""
    @tool("Test Tool", max_usage_count=3)
    def test_tool(input_text: str) -> str:
        """A test tool."""
        return f"Result: {input_text}"
    
    assert test_tool.max_usage_count == 3
    assert test_tool.current_usage_count == 0

    result = test_tool.run(input_text="test")
    assert result == "Result: test"
    assert test_tool.current_usage_count == 1


def test_default_unlimited_usage():
    """Test that tools have unlimited usage by default."""
    @tool("Default Tool")
    def default_tool(input_text: str) -> str:
        """A default tool."""
        return f"Result: {input_text}"
    
    assert default_tool.max_usage_count is None
    assert default_tool.current_usage_count == 0
