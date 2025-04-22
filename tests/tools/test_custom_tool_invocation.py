from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from crewai import Agent, Task
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentAction
from crewai.agents.tools_handler import ToolsHandler
from crewai.tools import BaseTool
from crewai.utilities.i18n import I18N
from crewai.utilities.tool_utils import execute_tool_and_check_finality


class TestToolInput(BaseModel):
    test_param: str = Field(..., description="A test parameter")


class TestCustomTool(BaseTool):
    name: str = "Test Custom Tool"
    description: str = "A test tool to verify custom tool invocation"
    args_schema: type[BaseModel] = TestToolInput

    def _run(self, test_param: str) -> str:
        return f"Tool executed with param: {test_param}"


def test_custom_tool_invocation():
    custom_tool = TestCustomTool()
    
    mock_agent = MagicMock()
    mock_task = MagicMock()
    mock_llm = MagicMock()
    mock_crew = MagicMock()
    tools_handler = ToolsHandler()
    
    executor = CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=mock_crew,
        agent=mock_agent,
        prompt={},
        max_iter=5,
        tools=[custom_tool],
        tools_names="Test Custom Tool",
        stop_words=[],
        tools_description="A test tool to verify custom tool invocation",
        tools_handler=tools_handler,
        original_tools=[custom_tool]
    )
    
    action = AgentAction(
        tool="Test Custom Tool",
        tool_input={"test_param": "test_value"},
        thought="I'll use the custom tool",
        text="I'll use the Test Custom Tool to get a result"
    )
    
    i18n = I18N()
    
    mock_agent.key = "test_agent"
    mock_agent.role = "test_role"
    
    result = execute_tool_and_check_finality(
        agent_action=action,
        tools=[custom_tool],
        i18n=i18n,
        agent_key=mock_agent.key,
        agent_role=mock_agent.role,
        tools_handler=tools_handler,
        task=mock_task,
        agent=mock_agent,
        function_calling_llm=mock_llm
    )
    
    assert "Tool executed with param: test_value" in result.result
    assert result.result_as_answer is False
