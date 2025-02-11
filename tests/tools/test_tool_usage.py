import json
import random
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from crewai import Agent, Task
from crewai.tools import BaseTool
from crewai.tools.tool_usage import ToolUsage


class RandomNumberToolInput(BaseModel):
    min_value: int = Field(
        ..., description="The minimum value of the range (inclusive)"
    )
    max_value: int = Field(
        ..., description="The maximum value of the range (inclusive)"
    )


class RandomNumberTool(BaseTool):
    name: str = "Random Number Generator"
    description: str = "Generates a random number within a specified range"
    args_schema: type[BaseModel] = RandomNumberToolInput

    def _run(self, min_value: int, max_value: int) -> int:
        return random.randint(min_value, max_value)


# Example agent and task
example_agent = Agent(
    role="Number Generator",
    goal="Generate random numbers for various purposes",
    backstory="You are an AI agent specialized in generating random numbers within specified ranges.",
    tools=[RandomNumberTool()],
    verbose=True,
)

example_task = Task(
    description="Generate a random number between 1 and 100",
    expected_output="A random number between 1 and 100",
    agent=example_agent,
)


def test_random_number_tool_range():
    tool = RandomNumberTool()
    result = tool._run(1, 10)
    assert 1 <= result <= 10


def test_random_number_tool_invalid_range():
    tool = RandomNumberTool()
    with pytest.raises(ValueError):
        tool._run(10, 1)  # min_value > max_value


def test_random_number_tool_schema():
    tool = RandomNumberTool()

    # Get the schema using model_json_schema()
    schema = tool.args_schema.model_json_schema()

    # Convert the schema to a string
    schema_str = json.dumps(schema)

    # Check if the schema string contains the expected fields
    assert "min_value" in schema_str
    assert "max_value" in schema_str

    # Parse the schema string back to a dictionary
    schema_dict = json.loads(schema_str)

    # Check if the schema contains the correct field types
    assert schema_dict["properties"]["min_value"]["type"] == "integer"
    assert schema_dict["properties"]["max_value"]["type"] == "integer"

    # Check if the schema contains the field descriptions
    assert (
        "minimum value" in schema_dict["properties"]["min_value"]["description"].lower()
    )
    assert (
        "maximum value" in schema_dict["properties"]["max_value"]["description"].lower()
    )


def test_tool_usage_interrupt_handling():
    """Test that tool usage properly propagates LangGraph interrupts."""
    from unittest.mock import patch, MagicMock

    class InterruptingTool(BaseTool):
        name: str = "interrupt_test"
        description: str = "A tool that raises LangGraph interrupts"

        def _run(self, query: str) -> str:
            raise type('Interrupt', (Exception,), {})("test interrupt")

    tool = InterruptingTool()
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[tool],
        original_tools=[tool],
        tools_description="Sample tool for testing",
        tools_names="interrupt_test",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    # Test that interrupt is propagated
    with pytest.raises(Exception) as exc_info:
        tool_usage.use(
            ToolCalling(tool_name="interrupt_test", arguments={"query": "test"}, log="test"),
            "test"
        )
    assert "test interrupt" in str(exc_info.value)

def test_tool_usage_render():
    tool = RandomNumberTool()

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[tool],
        original_tools=[tool],
        tools_description="Sample tool for testing",
        tools_names="random_number_generator",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    rendered = tool_usage._render()

    # Updated checks to match the actual output
    assert "Tool Name: Random Number Generator" in rendered
    assert "Tool Arguments:" in rendered
    assert (
        "'min_value': {'description': 'The minimum value of the range (inclusive)', 'type': 'int'}"
        in rendered
    )
    assert (
        "'max_value': {'description': 'The maximum value of the range (inclusive)', 'type': 'int'}"
        in rendered
    )
    assert (
        "Tool Description: Generates a random number within a specified range"
        in rendered
    )
    assert (
        "Tool Name: Random Number Generator\nTool Arguments: {'min_value': {'description': 'The minimum value of the range (inclusive)', 'type': 'int'}, 'max_value': {'description': 'The maximum value of the range (inclusive)', 'type': 'int'}}\nTool Description: Generates a random number within a specified range"
        in rendered
    )
