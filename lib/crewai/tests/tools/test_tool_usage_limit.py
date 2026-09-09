import pytest
from unittest.mock import MagicMock

from crewai.tools import BaseTool, tool
from crewai.tools.tool_calling import ToolCalling
from crewai.tools.tool_usage import ToolUsage


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


def test_invalid_usage_limit():
    """Test that negative usage limits raise ValueError."""
    class ValidTool(BaseTool):
        name: str = "Valid Tool"
        description: str = "A tool with valid usage limit"

        def _run(self, input_text: str) -> str:
            return f"Processed {input_text}"
    
    with pytest.raises(ValueError, match="max_usage_count must be a positive integer"):
        ValidTool(max_usage_count=-1)


def test_reset_usage_count():
    """Test that reset_usage_count method works correctly."""
    class LimitedTool(BaseTool):
        name: str = "Limited Tool"
        description: str = "A tool with usage limits for testing"
        max_usage_count: int = 3

        def _run(self, input_text: str) -> str:
            return f"Processed {input_text}"

    tool = LimitedTool()
    
    tool.run(input_text="test1")
    tool.run(input_text="test2")
    assert tool.current_usage_count == 2
    
    tool.reset_usage_count()
    assert tool.current_usage_count == 0
    
    result = tool.run(input_text="test3")
    assert result == "Processed test3"
    assert tool.current_usage_count == 1


def test_tool_usage_with_toolusage_class():
    """Test that ToolUsage class correctly enforces usage limits."""
    class LimitedTool(BaseTool):
        name: str = "Limited Tool"
        description: str = "A tool with usage limits for testing"
        max_usage_count: int = 2

        def _run(self, input_text: str) -> str:
            return f"Processed {input_text}"

    tool = LimitedTool()
    
    mock_agent = MagicMock()
    mock_task = MagicMock()
    mock_tools_handler = MagicMock()
    
    tool_usage = ToolUsage(
        tools=[tool],
        agent=mock_agent,
        task=mock_task,
        tools_handler=mock_tools_handler,
        function_calling_llm=MagicMock(),
    )
    
    tool_usage._check_tool_repeated_usage = MagicMock(return_value=False)
    tool_usage._format_result = lambda result: result
    
    mock_calling = MagicMock()
    mock_calling.tool_name = "Limited Tool"
    mock_calling.arguments = {"input_text": "test"}
    
    result1 = tool_usage._check_usage_limit(tool, "Limited Tool")
    assert result1 is None
    
    tool.current_usage_count += 1
    
    result2 = tool_usage._check_usage_limit(tool, "Limited Tool")
    assert result2 is None
    
    tool.current_usage_count += 1
    
    result3 = tool_usage._check_usage_limit(tool, "Limited Tool")
    assert "has reached its usage limit of 2 times" in result3


def test_tool_usage_use_does_not_double_count():
    """A tool run through ToolUsage.use() must increment its usage count once
    per call, not twice. Previously invoke() and ToolUsage._use both
    incremented, halving the effective max_usage_count."""
    runs = {"n": 0}

    class CountingTool(BaseTool):
        name: str = "counter"
        description: str = "counts"
        max_usage_count: int = 2

        def _run(self, x: int) -> str:
            runs["n"] += 1
            return f"got {x}"

    structured = CountingTool().to_structured_tool()

    tool_usage = ToolUsage(
        tools_handler=None,
        tools=[structured],
        task=None,
        function_calling_llm=None,
        agent=None,
        action=None,
    )

    tool_usage.use(calling=ToolCalling(tool_name="counter", arguments={"x": 1}), tool_string="")
    assert structured.current_usage_count == 1

    tool_usage.use(calling=ToolCalling(tool_name="counter", arguments={"x": 2}), tool_string="")
    assert structured.current_usage_count == 2

    # Both calls actually executed (the limit did not block the second one).
    assert runs["n"] == 2

    # The third call is over the limit and must not execute the tool.
    tool_usage.use(calling=ToolCalling(tool_name="counter", arguments={"x": 3}), tool_string="")
    assert runs["n"] == 2
    assert structured.current_usage_count == 2
