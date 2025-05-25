"""Test for A2A protocol support in CrewAI."""

import pytest
from unittest.mock import patch, MagicMock

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.tools.agent_tools.delegate_work_tool import DelegateWorkTool
from crewai.tools.agent_tools.ask_question_tool import AskQuestionTool
from crewai.tools.base_tool import BaseTool


def test_tools_passed_to_execute():
    """Test that tools are properly passed to the _execute method."""
    original_execute = DelegateWorkTool._execute
    
    tools_passed = {"value": False}
    
    def mock_execute(self, agent_name, task, context=None, tools=None):
        assert tools is not None, "Tools should not be None"
        assert len(tools) > 0, "Tools should not be empty"
        assert any(isinstance(tool, DelegateWorkTool) for tool in tools), "DelegateWorkTool should be in tools"
        assert any(isinstance(tool, AskQuestionTool) for tool in tools), "AskQuestionTool should be in tools"
        tools_passed["value"] = True
        return "Task executed successfully"
    
    researcher = Agent(
        role="researcher",
        goal="research and analyze content",
        backstory="You're an expert researcher",
        allow_delegation=True,
    )
    
    writer = Agent(
        role="writer",
        goal="write content based on research",
        backstory="You're an expert writer",
        allow_delegation=True,
    )
    
    agent_tools = AgentTools(agents=[researcher, writer])
    delegation_tools = agent_tools.tools()
    
    with patch.object(DelegateWorkTool, '_execute', mock_execute):
        delegate_tool = delegation_tools[0]  # DelegateWorkTool is the first tool
        assert isinstance(delegate_tool, DelegateWorkTool), "First tool should be DelegateWorkTool"
        
        delegate_tool._run(
            task="Test task",
            context="Test context",
            coworker="writer",
            tools=delegation_tools
        )
        
        assert tools_passed["value"], "Tools should be passed to _execute method"


def test_tools_passed_from_ask_question_tool():
    """Test that tools are properly passed from AskQuestionTool to _execute."""
    original_execute = AskQuestionTool._execute
    
    tools_passed = {"value": False}
    
    def mock_execute(self, agent_name, question, context=None, tools=None):
        assert tools is not None, "Tools should not be None"
        assert len(tools) > 0, "Tools should not be empty"
        assert any(isinstance(tool, DelegateWorkTool) for tool in tools), "DelegateWorkTool should be in tools"
        assert any(isinstance(tool, AskQuestionTool) for tool in tools), "AskQuestionTool should be in tools"
        tools_passed["value"] = True
        return "Question answered successfully"
    
    researcher = Agent(
        role="researcher",
        goal="research and analyze content",
        backstory="You're an expert researcher",
        allow_delegation=True,
    )
    
    writer = Agent(
        role="writer",
        goal="write content based on research",
        backstory="You're an expert writer",
        allow_delegation=True,
    )
    
    agent_tools = AgentTools(agents=[researcher, writer])
    delegation_tools = agent_tools.tools()
    
    with patch.object(AskQuestionTool, '_execute', mock_execute):
        ask_tool = delegation_tools[1]  # AskQuestionTool is the second tool
        assert isinstance(ask_tool, AskQuestionTool), "Second tool should be AskQuestionTool"
        
        ask_tool._run(
            question="Test question",
            context="Test context",
            coworker="writer",
            tools=delegation_tools
        )
        
        assert tools_passed["value"], "Tools should be passed to _execute method"


def test_agent_tools_injects_tools():
    """Test that AgentTools injects tools into delegation tools."""
    researcher = Agent(
        role="researcher",
        goal="research and analyze content",
        backstory="You're an expert researcher",
        allow_delegation=True,
    )
    
    writer = Agent(
        role="writer",
        goal="write content based on research",
        backstory="You're an expert writer",
        allow_delegation=True,
    )
    
    class CustomTool(BaseTool):
        name: str = "Custom Tool"
        description: str = "A custom tool for testing"
        
        def _run(self, *args, **kwargs):
            return "Custom tool executed"
    
    custom_tool = CustomTool()
    researcher.tools = [custom_tool]
    
    agent_tools = AgentTools(agents=[researcher, writer])
    delegation_tools = agent_tools.tools()
    
    for tool in delegation_tools:
        assert hasattr(tool, '_agent_tools'), "Tool should have _agent_tools attribute"
        assert len(tool._agent_tools) > 0, "Tool should have agent tools injected"
        assert any(isinstance(t, CustomTool) for t in tool._agent_tools), "Custom tool should be injected"
