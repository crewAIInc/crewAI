from collections.abc import Callable
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool


# Test fixtures
@pytest.fixture
def basic_function() -> Callable[[str, int], str]:
    def test_func(param1: str, param2: int = 0) -> str:
        """Test function with basic params."""
        return f"{param1} {param2}"

    return test_func


@pytest.fixture
def schema_class() -> type[BaseModel]:
    class TestSchema(BaseModel):
        param1: str
        param2: int = Field(default=0)

    return TestSchema


def test_initialization(
    basic_function: Callable[[str], str], schema_class: type[BaseModel]
) -> None:
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


def test_from_function(basic_function: Callable[[str], str]) -> None:
    """Test creating tool from function"""
    tool = CrewStructuredTool.from_function(
        func=basic_function, name="test_tool", description="Test description"
    )

    assert tool.name == "test_tool"
    assert tool.description == "Test description"
    assert tool.func == basic_function
    assert isinstance(tool.args_schema, type(BaseModel))


def test_validate_function_signature(
    basic_function: Callable[[str, int], str], schema_class: type[BaseModel]
) -> None:
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
async def test_ainvoke(basic_function: Callable[[str, int], str]) -> None:
    """Test asynchronous invocation"""
    tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

    result = await tool.ainvoke(input={"param1": "test"})
    assert result == "test 0"


def test_parse_args_dict(basic_function: Callable[[str, int], str]) -> None:
    """Test parsing dictionary arguments"""
    tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

    parsed = tool._parse_args({"param1": "test", "param2": 42})
    assert parsed["param1"] == "test"
    assert parsed["param2"] == 42


def test_parse_args_string(basic_function: Callable[[str, int], str]) -> None:
    """Test parsing string arguments"""
    tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

    parsed = tool._parse_args('{"param1": "test", "param2": 42}')
    assert parsed["param1"] == "test"
    assert parsed["param2"] == 42


def test_complex_types() -> None:
    """Test handling of complex parameter types"""

    def complex_func(nested: dict[str, Any], items: list[Any]) -> str:
        """Process complex types."""
        return f"Processed {len(items)} items with {len(nested)} nested keys"

    tool = CrewStructuredTool.from_function(
        func=complex_func, name="test_tool", description="Test complex types"
    )
    result = tool.invoke({"nested": {"key": "value"}, "items": [1, 2, 3]})
    assert result == "Processed 3 items with 1 nested keys"


def test_schema_inheritance() -> None:
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


def test_default_values_in_schema() -> None:
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
def custom_tool_decorator() -> Any:
    from crewai.tools import tool

    @tool("custom_tool", result_as_answer=True)
    async def custom_tool() -> str:
        """This is a tool that does something"""
        return "Hello World from Custom Tool"

    return custom_tool


@pytest.fixture
def custom_tool() -> BaseTool:
    from crewai.tools import BaseTool

    class CustomTool(BaseTool):
        name: str = "my_tool"
        description: str = "This is a tool that does something"
        result_as_answer: bool = True

        async def _run(self) -> str:
            return "Hello World from Custom Tool"

    return CustomTool()


def build_simple_crew(tool: Any) -> Any:
    from crewai import Agent, Crew, Task

    agent1 = Agent(
        role="Simple role",
        goal="Simple goal",
        backstory="Simple backstory",
        tools=[tool],
    )

    say_hi_task = Task(
        description="Use the custom tool result as answer.",
        agent=agent1,
        expected_output="Use the tool result",
    )

    crew = Crew(agents=[agent1], tasks=[say_hi_task])
    return crew


@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_using_within_isolated_crew(custom_tool: BaseTool) -> None:
    crew = build_simple_crew(custom_tool)
    result = crew.kickoff()

    assert result.raw == "Hello World from Custom Tool"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_using_decorator_within_isolated_crew(
    custom_tool_decorator: Any,
) -> None:
    crew = build_simple_crew(custom_tool_decorator)
    result = crew.kickoff()

    assert result.raw == "Hello World from Custom Tool"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_within_flow(custom_tool: BaseTool) -> None:
    from crewai.flow.flow import Flow, start

    class StructuredExampleFlow(Flow):  # type: ignore[type-arg]
        @start()
        async def start(self) -> Any:
            crew = build_simple_crew(custom_tool)
            result = await crew.kickoff_async()
            return result

    flow = StructuredExampleFlow()
    result = flow.kickoff()
    assert result.raw == "Hello World from Custom Tool"


@pytest.mark.vcr(filter_headers=["authorization"])
def test_async_tool_using_decorator_within_flow(custom_tool_decorator: Any) -> None:
    from crewai.flow.flow import Flow, start

    class StructuredExampleFlow(Flow):  # type: ignore[type-arg]
        @start()
        async def start(self) -> Any:
            crew = build_simple_crew(custom_tool_decorator)
            result = await crew.kickoff_async()
            return result

    flow = StructuredExampleFlow()
    result = flow.kickoff()
    assert result.raw == "Hello World from Custom Tool"


def test_invoke_sync_function_single_execution() -> None:
    """Test that sync functions are called only once, not twice."""
    call_count = 0

    def counting_func(message: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Called {call_count} times with: {message}"

    tool = CrewStructuredTool.from_function(
        func=counting_func, name="counting_tool", description="A tool that counts calls"
    )

    result = tool.invoke({"message": "test"})
    assert call_count == 1, f"Function was called {call_count} times, expected 1"
    assert result == "Called 1 times with: test"


def test_invoke_async_function_outside_event_loop() -> None:
    """Test that async functions work correctly when called outside event loop."""

    async def async_func(message: str) -> str:
        return f"Async result: {message}"

    tool = CrewStructuredTool.from_function(
        func=async_func, name="async_tool", description="An async tool"
    )

    result = tool.invoke({"message": "test"})
    assert result == "Async result: test"


@pytest.mark.asyncio
async def test_invoke_async_function_in_event_loop_raises_error() -> None:
    """Test that async functions raise RuntimeError when called from within event loop."""

    async def async_func(message: str) -> str:
        return f"Async result: {message}"

    tool = CrewStructuredTool.from_function(
        func=async_func, name="async_tool", description="An async tool"
    )

    with pytest.raises(
        RuntimeError,
        match="Cannot call async tool.*from synchronous context within an event loop",
    ):
        tool.invoke({"message": "test"})


def test_invoke_sync_function_returning_coroutine() -> None:
    """Test handling of sync functions that return coroutines."""

    async def inner_async(message: str) -> str:
        return f"Inner async: {message}"

    def sync_func_returning_coro(message: str) -> Any:
        return inner_async(message)

    tool = CrewStructuredTool.from_function(
        func=sync_func_returning_coro,
        name="sync_coro_tool",
        description="A sync tool that returns coroutine",
    )

    result = tool.invoke({"message": "test"})
    assert result == "Inner async: test"


@pytest.mark.asyncio
async def test_invoke_sync_function_returning_coroutine_in_event_loop_raises_error() -> (
    None
):
    """Test that sync functions returning coroutines raise RuntimeError in event loop."""

    async def inner_async(message: str) -> str:
        return f"Inner async: {message}"

    def sync_func_returning_coro(message: str) -> Any:
        return inner_async(message)

    tool = CrewStructuredTool.from_function(
        func=sync_func_returning_coro,
        name="sync_coro_tool",
        description="A sync tool that returns coroutine",
    )

    with pytest.raises(
        RuntimeError,
        match="Sync function.*returned a coroutine but we're in an event loop",
    ):
        tool.invoke({"message": "test"})
