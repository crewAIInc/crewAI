"""
Test custom tool registration patterns to ensure all documented patterns work correctly.
This addresses issue #3226 where custom tool registration was broken in CrewAI 0.150.0.
"""

import pytest
from typing import Any
from crewai import Agent
from crewai.tools import BaseTool, tool, Tool


class TestCustomToolPatterns:
    """Test all custom tool patterns mentioned in issue #3226."""

    def test_function_tool_with_decorator(self):
        """Test function tool with @tool decorator."""
        @tool
        def fetch_logs(query: str) -> str:
            """Fetch logs from New Relic based on query"""
            return f"Logs for query: {query}"
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[fetch_logs],
            allow_delegation=False
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "fetch_logs"
        assert "Fetch logs from New Relic" in agent.tools[0].description

    def test_dict_based_tool(self):
        """Test dict-based tool definition."""
        def fetch_logs_func(query: str) -> str:
            return f"Logs for query: {query}"
        
        fetch_logs_dict = {
            'name': 'fetch_logs',
            'description': 'Fetch logs from New Relic',
            'func': fetch_logs_func
        }
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[fetch_logs_dict],
            allow_delegation=False
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "fetch_logs"
        assert "Fetch logs from New Relic" in agent.tools[0].description

    def test_basetool_class_inheritance(self):
        """Test BaseTool class inheritance."""
        class FetchLogsTool(BaseTool):
            name: str = "fetch_logs"
            description: str = "Fetch logs from New Relic based on query"
            
            def _run(self, query: str) -> str:
                return f"Logs for query: {query}"
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[FetchLogsTool()],
            allow_delegation=False
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "fetch_logs"
        assert "Fetch logs from New Relic" in agent.tools[0].description

    def test_direct_function_assignment(self):
        """Test direct function assignment."""
        def fetch_logs(query: str) -> str:
            """Fetch logs from New Relic based on query"""
            return f"Logs for query: {query}"
        
        agent = Agent(
            role='CrashFetcher',
            goal='Extract logs',
            backstory='An agent that fetches logs',
            tools=[fetch_logs],
            allow_delegation=False
        )
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "fetch_logs"
        assert "Fetch logs from New Relic" in agent.tools[0].description

    def test_mixed_tool_types(self):
        """Test mixing different tool types in the same agent."""
        @tool
        def decorated_tool(query: str) -> str:
            """A decorated tool"""
            return f"Decorated: {query}"
        
        class ClassTool(BaseTool):
            name: str = "class_tool"
            description: str = "A class-based tool"
            
            def _run(self, query: str) -> str:
                return f"Class: {query}"
        
        def function_tool(query: str) -> str:
            """A function tool"""
            return f"Function: {query}"
        
        dict_tool = {
            'name': 'dict_tool',
            'description': 'A dict-based tool',
            'func': lambda query: f"Dict: {query}"
        }
        
        agent = Agent(
            role='MultiTool',
            goal='Use multiple tool types',
            backstory='An agent with various tools',
            tools=[decorated_tool, ClassTool(), function_tool, dict_tool],
            allow_delegation=False
        )
        
        assert len(agent.tools) == 4
        tool_names = [tool.name for tool in agent.tools]
        assert "decorated_tool" in tool_names
        assert "class_tool" in tool_names
        assert "function_tool" in tool_names
        assert "dict_tool" in tool_names

    def test_invalid_tool_types(self):
        """Test that invalid tool types raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid tool type"):
            Agent(
                role='Test',
                goal='Test invalid tools',
                backstory='Testing',
                tools=["invalid_string_tool"],
                allow_delegation=False
            )
        
        with pytest.raises(ValueError, match="Invalid tool type"):
            Agent(
                role='Test',
                goal='Test invalid tools',
                backstory='Testing',
                tools=[123],
                allow_delegation=False
            )

    def test_function_without_docstring_fails(self):
        """Test that functions without docstrings fail validation."""
        def no_docstring_func(query: str) -> str:
            return f"No docstring: {query}"
        
        with pytest.raises(ValueError, match="must have a docstring"):
            Agent(
                role='Test',
                goal='Test function without docstring',
                backstory='Testing',
                tools=[no_docstring_func],
                allow_delegation=False
            )

    def test_incomplete_dict_tool_fails(self):
        """Test that dict tools missing required keys fail validation."""
        incomplete_dict = {
            'name': 'incomplete',
            'description': 'Missing func key'
        }
        
        with pytest.raises(ValueError, match="Dict tool must contain keys"):
            Agent(
                role='Test',
                goal='Test incomplete dict tool',
                backstory='Testing',
                tools=[incomplete_dict],
                allow_delegation=False
            )

    def test_tool_execution(self):
        """Test that tools can actually be executed."""
        @tool
        def test_execution_tool(message: str) -> str:
            """A tool for testing execution"""
            return f"Executed: {message}"
        
        agent = Agent(
            role='Executor',
            goal='Execute tools',
            backstory='An agent that executes tools',
            tools=[test_execution_tool],
            allow_delegation=False
        )
        
        tool_instance = agent.tools[0]
        result = tool_instance.run(message="test")
        assert result == "Executed: test"

    def test_tool_with_multiple_parameters(self):
        """Test tools with multiple parameters work correctly."""
        @tool
        def multi_param_tool(param1: str, param2: int, param3: bool = True) -> str:
            """A tool with multiple parameters"""
            return f"p1={param1}, p2={param2}, p3={param3}"
        
        agent = Agent(
            role='MultiParam',
            goal='Use multi-parameter tools',
            backstory='An agent with complex tools',
            tools=[multi_param_tool],
            allow_delegation=False
        )
        
        assert len(agent.tools) == 1
        tool_instance = agent.tools[0]
        result = tool_instance.run(param1="test", param2=42, param3=False)
        assert result == "p1=test, p2=42, p3=False"
