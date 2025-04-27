import pytest
from pydantic import BaseModel, Field

from crewai.task import Task
from crewai.tools.structured_tool import CrewStructuredTool


@pytest.fixture
def simple_tool_function():
    def test_func(param1: str, param2: int = 0) -> str:
        """Test function with basic params."""
        return f"{param1} {param2}"

    return test_func


def test_task_with_structured_tool(simple_tool_function):
    """Test that CrewStructuredTool can be used directly with Task."""
    tool = CrewStructuredTool.from_function(
        func=simple_tool_function,
        name="test_tool",
        description="Test tool description"
    )
    
    task = Task(
        description="Test task description",
        expected_output="Expected output",
        tools=[tool]
    )
    
    assert len(task.tools) == 1
    assert task.tools[0] == tool


def test_mixed_tool_types(simple_tool_function):
    """Test that both BaseTool and CrewStructuredTool can be used together with Task."""
    from crewai.tools import BaseTool
    
    structured_tool = CrewStructuredTool.from_function(
        func=simple_tool_function,
        name="structured_tool",
        description="Structured tool description"
    )
    
    class TestBaseTool(BaseTool):
        name: str = "base_tool"
        description: str = "Base tool description"
        
        def _run(self, query: str) -> str:
            return f"Result for {query}"
    
    base_tool = TestBaseTool()
    
    task = Task(
        description="Test task description",
        expected_output="Expected output",
        tools=[structured_tool, base_tool]
    )
    
    assert len(task.tools) == 2
    assert task.tools[0] == structured_tool
    assert isinstance(task.tools[1], BaseTool)
