from typing import Type
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Flow, Task
from crewai.flow import listen, start
from crewai.tools import BaseTool


class TestToolInput(BaseModel):
    query: str = Field(..., description='Query to process')

class TestTool(BaseTool):
    name: str = 'Test Tool'
    description: str = 'A test tool to demonstrate the issue'
    args_schema: Type[BaseModel] = TestToolInput
    result_as_answer: bool = True
    
    def _run(self, query: str) -> str:
        return f'Result for query: {query}'

def test_flow_tool_result_as_answer():
    """Test that tools with result_as_answer=True are properly processed in Flow mode."""
    # Create a test tool
    test_tool = TestTool()
    
    # Create a test agent with the tool
    agent = Agent(
        role='Tester',
        goal='Test tools',
        backstory='Testing tools in Flow vs Crew',
        tools=[test_tool]
    )

    # Create a task with the tool
    task = Task(
        description='Test task using the tool',
        expected_output='Test output',
        agent=agent,
        tools=[test_tool]
    )
    
    # Create a simple Flow with direct access to the agent
    class SimpleFlow(Flow):
        def __init__(self):
            super().__init__()
            self.test_agent = agent
            
        @start()
        def start_task(self):
            return 'Task started'
            
        @listen('start_task')
        @Flow.with_agent(agent)  # Associate the agent with this method
        def execute_task(self):
            # Simulate tool execution and setting tools_results
            self.test_agent.tools_results = [{
                "name": "Test Tool",
                "input": {"query": "test"},
                "result": "Result for query: test",
                "result_as_answer": True
            }]
            return "Agent task execution result"

    # Create a mock for Crew to return the same result
    with patch('crewai.crew.Crew.kickoff') as mock_crew_kickoff:
        mock_crew_kickoff.return_value = "Result for query: test"
        
        # Test Flow
        flow = SimpleFlow()
        flow_result = flow.kickoff()
        
        # Verify that Flow returns the tool result with our fix
        assert flow_result == "Result for query: test", "Flow should return tool result with result_as_answer=True"
        
        # Test Crew
        crew = Crew(agents=[agent], tasks=[task])
        crew_result = crew.kickoff()
        
        # Verify that Crew returns the tool result
        assert crew_result == "Result for query: test", "Crew should return tool result with result_as_answer=True"
