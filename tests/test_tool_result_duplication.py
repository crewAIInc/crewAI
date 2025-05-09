import pytest
from typing import Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.tools_handler import ToolsHandler
from crewai.crew import Crew
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.agent_utils import parse_tools


class TestTool(BaseTool):
    name: str = "Test Tool"
    description: str = "A test tool to verify tool result duplication is fixed"

    def _run(self) -> str:
        return "Test tool result"


def test_tool_result_not_duplicated_in_messages() -> None:
    """Test that tool results are not duplicated in messages.
    
    This test verifies the fix for issue #2798, where tool results were being
    duplicated in the LLM prompt, increasing token usage and latency.
    """
    agent = Agent(
        role="Test Agent",
        goal="Test the tool result duplication fix",
        backstory="I am a test agent",
        tools=[TestTool()],
    )
    
    task = Task(
        description="Use the test tool and return the result",
        expected_output="The test tool result",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    structured_tools = parse_tools(agent.tools)
    tools_names = ", ".join([t.name for t in structured_tools])
    tools_description = "\n".join([t.description for t in structured_tools])
    
    tools_handler = MagicMock(spec=ToolsHandler)
    
    with patch.object(CrewAgentExecutor, '_invoke_loop') as mock_invoke_loop:
        executor = CrewAgentExecutor(
            agent=agent,
            task=task,
            crew=crew,
            llm=agent.llm,
            prompt={"system": "System prompt", "user": "User prompt"},
            max_iter=10,
            tools=structured_tools,
            tools_names=tools_names,
            stop_words=[],
            tools_description=tools_description,
            tools_handler=tools_handler,
            callbacks=[],
        )
        
        executor.messages = [{"role": "user", "content": "Use the test tool"}]
        
        from crewai.agents.parser import AgentAction
        from crewai.tools.tool_types import ToolResult
        
        agent_action = AgentAction(
            tool="Test Tool",
            tool_input={},
            thought="I should use the test tool",
            text="I'll use the Test Tool",
        )
        
        tool_result = ToolResult(
            result="Test tool result",
            tool_name="Test Tool",
            tool_args={},
            result_as_answer=False,
        )
        
        executor._handle_agent_action(agent_action, tool_result)
        
        tool_result_count = sum(
            1 for msg in executor.messages if msg.get("content") == "Test tool result"
        )
        
        assert tool_result_count <= 1, "Tool result is duplicated in messages"
        
        observation_text = f"Observation: {tool_result.result}"
        assert observation_text in agent_action.text, "Tool result not properly formatted in agent action text"
