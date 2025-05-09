import pytest
from unittest.mock import patch, MagicMock

from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew
from crewai.tools import BaseTool
from crewai.agents.crew_agent_executor import CrewAgentExecutor


class TestTool(BaseTool):
    name: str = "Test Tool"
    description: str = "A test tool to verify tool result duplication is fixed"

    def _run(self) -> str:
        return "Test tool result"


def test_tool_result_not_duplicated_in_messages():
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
    
    with patch.object(CrewAgentExecutor, '_invoke_loop') as mock_invoke_loop:
        executor = CrewAgentExecutor(
            agent=agent,
            task=task,
            tools=agent.tools,
            llm=agent.llm,
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
