import pytest
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

from crewai import Agent, Crew, Task
from crewai.agents.tools_handler import ToolsHandler
from crewai.tasks.task_output import TaskOutput
from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_calling import InstructorToolCalling, ToolCalling
from crewai.tools.tool_usage import ToolUsage


class TestCodeInterpreterTool(BaseTool):
    name: str = "Test Code Interpreter"
    description: str = "A test tool that simulates code execution."
    result_as_answer: bool = False
    execution_called: bool = False
    
    def _run(self, code: str = "", libraries_used: List[str] = []) -> str:
        self.execution_called = True
        return f"Code executed: {code}"


def test_direct_tool_execution():
    """Test that the tool can be executed directly."""
    test_tool = TestCodeInterpreterTool()
    result = test_tool.run("print('Hello World')")
    assert "Code executed" in result
    assert test_tool.execution_called




def test_tool_usage_return_types():
    """Test that the ToolUsage methods return the correct types."""
    agent_mock = MagicMock()
    task_mock = MagicMock()
    task_mock.used_tools = 0
    tools_handler_mock = MagicMock()
    
    tool_usage = ToolUsage(
        tools_handler=tools_handler_mock,
        tools=[],
        task=task_mock,
        function_calling_llm=None,
        agent=agent_mock,
        action=None
    )
    
    i18n_mock = MagicMock()
    i18n_mock.slice.return_value.format.return_value = "Tool description"
    tool_usage._i18n = i18n_mock
    
    result = "test result"
    formatted_result = tool_usage._format_result(result)
    assert formatted_result is not None, "_format_result should return the result"
    assert isinstance(formatted_result, str), "_format_result should return a string"
    
    result = tool_usage._should_remember_format()
    assert isinstance(result, bool), "_should_remember_format should return a boolean"
    
    result = "test result"
    remembered_result = tool_usage._remember_format(result)
    assert remembered_result is not None, "_remember_format should return the result"
    assert isinstance(remembered_result, str), "_remember_format should return a string"
    
    tool_calling_mock = MagicMock()
    result = tool_usage._check_tool_repeated_usage(tool_calling_mock)
    assert isinstance(result, bool), "_check_tool_repeated_usage should return a boolean"
