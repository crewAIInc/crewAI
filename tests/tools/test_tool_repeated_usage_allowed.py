import pytest
from unittest.mock import MagicMock

from crewai.tools import BaseTool
from crewai.tools.tool_calling import ToolCalling
from crewai.tools.tool_usage import ToolUsage


def test_tool_repeated_usage_allowed():
    """Test that a tool with allow_repeated_usage=True can be used repeatedly with same args."""
    
    class RepeatedUsageTool(BaseTool):
        name: str = "Repeated Usage Tool"
        description: str = "A tool that can be used repeatedly with the same arguments"
        allow_repeated_usage: bool = True
        
        def _run(self, test_arg: str) -> str:
            return f"Used with arg: {test_arg}"
    
    # Setup tool usage
    tool = RepeatedUsageTool()
    tools_handler = MagicMock()
    tools_handler.last_used_tool = ToolCalling(
        tool_name="Repeated Usage Tool",
        arguments={"test_arg": "test"}
    )
    
    tool_usage = ToolUsage(
        tools_handler=tools_handler,
        tools=[tool],
        original_tools=[tool],
        tools_description="Test tools",
        tools_names="Repeated Usage Tool",
        agent=MagicMock(),
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        action=MagicMock(),
    )
    
    # Create a new tool calling with the same arguments
    calling = ToolCalling(
        tool_name="Repeated Usage Tool",
        arguments={"test_arg": "test"}
    )
    
    # This should return False since the tool allows repeated usage
    result = tool_usage._check_tool_repeated_usage(calling=calling)
    assert result is False
