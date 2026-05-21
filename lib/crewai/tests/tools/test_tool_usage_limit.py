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


def test_tool_usage_increments_structured_tool_once_per_call():
    """Test that agent tool execution only consumes one usage per invocation."""

    class LimitedTool(BaseTool):
        name: str = "Limited Tool"
        description: str = "A tool with usage limits for testing"
        max_usage_count: int = 2

        def _run(self, input_text: str) -> str:
            return f"Processed {input_text}"

    original_tool = LimitedTool()
    structured_tool = original_tool.to_structured_tool()
    llm = MagicMock()
    llm.model = "gpt-4o-mini"
    tool_usage = ToolUsage(
        tools=[structured_tool],
        agent=None,
        task=None,
        tools_handler=None,
        function_calling_llm=llm,
    )

    first_result = tool_usage.use(
        calling=ToolCalling(
            tool_name="Limited Tool", arguments={"input_text": "first"}
        ),
        tool_string="",
    )
    assert first_result == "Processed first"
    assert structured_tool.current_usage_count == 1
    assert original_tool.current_usage_count == 1

    second_result = tool_usage.use(
        calling=ToolCalling(
            tool_name="Limited Tool", arguments={"input_text": "second"}
        ),
        tool_string="",
    )
    assert second_result == "Processed second"
    assert structured_tool.current_usage_count == 2
    assert original_tool.current_usage_count == 2

    third_result = tool_usage.use(
        calling=ToolCalling(
            tool_name="Limited Tool", arguments={"input_text": "third"}
        ),
        tool_string="",
    )

    assert "has reached its usage limit of 2 times" in third_result
    assert structured_tool.current_usage_count == 2
    assert original_tool.current_usage_count == 2


@pytest.mark.asyncio
async def test_tool_usage_increments_structured_tool_once_per_async_call():
    """Test that async agent tool execution only consumes one usage per invocation."""

    class LimitedTool(BaseTool):
        name: str = "Limited Tool"
        description: str = "A tool with usage limits for testing"
        max_usage_count: int = 2

        def _run(self, input_text: str) -> str:
            return f"Processed {input_text}"

    original_tool = LimitedTool()
    structured_tool = original_tool.to_structured_tool()
    llm = MagicMock()
    llm.model = "gpt-4o-mini"
    tool_usage = ToolUsage(
        tools=[structured_tool],
        agent=None,
        task=None,
        tools_handler=None,
        function_calling_llm=llm,
    )

    first_result = await tool_usage.ause(
        calling=ToolCalling(
            tool_name="Limited Tool", arguments={"input_text": "first"}
        ),
        tool_string="",
    )
    assert first_result == "Processed first"
    assert structured_tool.current_usage_count == 1
    assert original_tool.current_usage_count == 1

    second_result = await tool_usage.ause(
        calling=ToolCalling(
            tool_name="Limited Tool", arguments={"input_text": "second"}
        ),
        tool_string="",
    )
    assert second_result == "Processed second"
    assert structured_tool.current_usage_count == 2
    assert original_tool.current_usage_count == 2

    third_result = await tool_usage.ause(
        calling=ToolCalling(
            tool_name="Limited Tool", arguments={"input_text": "third"}
        ),
        tool_string="",
    )

    assert "has reached its usage limit of 2 times" in third_result
    assert structured_tool.current_usage_count == 2
    assert original_tool.current_usage_count == 2
