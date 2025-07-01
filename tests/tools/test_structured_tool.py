from typing import Optional

import pytest
from pydantic import BaseModel, Field

from crewai.tools.structured_tool import CrewStructuredTool


# Test fixtures
@pytest.fixture
def basic_function():
    def test_func(param1: str, param2: int = 0) -> str:
        """Test function with basic params."""
        return f"{param1} {param2}"

    return test_func


@pytest.fixture
def schema_class():
    class TestSchema(BaseModel):
        param1: str
        param2: int = Field(default=0)

    return TestSchema


def test_initialization(basic_function, schema_class):
    """Test basic initialization of CrewStructuredTool"""
    tool = CrewStructuredTool(
        name="test_tool",
        description="Test tool description",
        func=basic_function,
        args_schema=schema_class,
    )

    assert tool.name == "test_tool"
    assert tool.description == "Test tool description"
    assert tool.func == basic_function
    assert tool.args_schema == schema_class

def test_from_function(basic_function):
    """Test creating tool from function"""
    tool = CrewStructuredTool.from_function(
        func=basic_function, name="test_tool", description="Test description"
    )

    assert tool.name == "test_tool"
    assert tool.description == "Test description"
    assert tool.func == basic_function
    assert isinstance(tool.args_schema, type(BaseModel))

def test_validate_function_signature(basic_function, schema_class):
    """Test function signature validation"""
    tool = CrewStructuredTool(
        name="test_tool",
        description="Test tool",
        func=basic_function,
        args_schema=schema_class,
    )

    # Should not raise any exceptions
    tool._validate_function_signature()

@pytest.mark.asyncio
async def test_ainvoke(basic_function):
    """Test asynchronous invocation"""
    tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

    result = await tool.ainvoke(input={"param1": "test"})
    assert result == "test 0"

def test_parse_args_dict(basic_function):
    """Test parsing dictionary arguments"""
    tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

    parsed = tool._parse_args({"param1": "test", "param2": 42})
    assert parsed["param1"] == "test"
    assert parsed["param2"] == 42

def test_parse_args_string(basic_function):
    """Test parsing string arguments"""
    tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

    parsed = tool._parse_args('{"param1": "test", "param2": 42}')
    assert parsed["param1"] == "test"
    assert parsed["param2"] == 42

def test_complex_types():
    """Test handling of complex parameter types"""

    def complex_func(nested: dict, items: list) -> str:
        """Process complex types."""
        return f"Processed {len(items)} items with {len(nested)} nested keys"

    tool = CrewStructuredTool.from_function(
        func=complex_func, name="test_tool", description="Test complex types"
    )
    result = tool.invoke({"nested": {"key": "value"}, "items": [1, 2, 3]})
    assert result == "Processed 3 items with 1 nested keys"

def test_schema_inheritance():
    """Test tool creation with inherited schema"""

    def extended_func(base_param: str, extra_param: int) -> str:
        """Test function with inherited schema."""
        return f"{base_param} {extra_param}"

    class BaseSchema(BaseModel):
        base_param: str

    class ExtendedSchema(BaseSchema):
        extra_param: int

    tool = CrewStructuredTool.from_function(
        func=extended_func, name="test_tool", args_schema=ExtendedSchema
    )

    result = tool.invoke({"base_param": "test", "extra_param": 42})
    assert result == "test 42"

def test_default_values_in_schema():
    """Test handling of default values in schema"""

    def default_func(
        required_param: str,
        optional_param: str = "default",
        nullable_param: Optional[int] = None,
    ) -> str:
        """Test function with default values."""
        return f"{required_param} {optional_param} {nullable_param}"

    tool = CrewStructuredTool.from_function(
        func=default_func, name="test_tool", description="Test defaults"
    )

    # Test with minimal parameters
    result = tool.invoke({"required_param": "test"})
    assert result == "test default None"

    # Test with all parameters
    result = tool.invoke(
        {"required_param": "test", "optional_param": "custom", "nullable_param": 42}
    )
    assert result == "test custom 42"

@pytest.fixture
def custom_tool_decorator():
    from crewai.tools import tool

    @tool("custom_tool", result_as_answer=True)
    async def custom_tool():
        """This is a tool that does something"""
        return "Hello World from Custom Tool"

    return custom_tool

@pytest.fixture
def custom_tool():
    from crewai.tools import BaseTool

    class CustomTool(BaseTool):
        name: str = "my_tool"
        description: str = "This is a tool that does something"
        result_as_answer: bool = True

        async def _run(self):
            return "Hello World from Custom Tool"

    return CustomTool()

def build_simple_crew(tool):
    from crewai import Agent, Task, Crew

    agent1 = Agent(role="Simple role", goal="Simple goal", backstory="Simple backstory", tools=[tool])

    say_hi_task = Task(
        description="Use the custom tool result as answer.", agent=agent1, expected_output="Use the tool result"
    )

    crew = Crew(agents=[agent1], tasks=[say_hi_task])
    return crew

@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_using_within_isolated_crew(custom_tool):
    crew = build_simple_crew(custom_tool)
    result = crew.kickoff()

    assert result.raw == "Hello World from Custom Tool"

@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_using_decorator_within_isolated_crew(custom_tool_decorator):
    crew = build_simple_crew(custom_tool_decorator)
    result = crew.kickoff()

    assert result.raw == "Hello World from Custom Tool"

@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_within_flow(custom_tool):
    from crewai.flow.flow import Flow

    class StructuredExampleFlow(Flow):
        from crewai.flow.flow import start

        @start()
        async def start(self):
            crew = build_simple_crew(custom_tool)
            result = await crew.kickoff_async()
            return result

    flow = StructuredExampleFlow()
    result = flow.kickoff()
    assert result.raw == "Hello World from Custom Tool"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_using_decorator_within_flow(custom_tool_decorator):
    from crewai.flow.flow import Flow

    class StructuredExampleFlow(Flow):
        from crewai.flow.flow import start
        @start()
        async def start(self):
            crew = build_simple_crew(custom_tool_decorator)
            result = await crew.kickoff_async()
            return result

    flow = StructuredExampleFlow()
    result = flow.kickoff()
    assert result.raw == "Hello World from Custom Tool"