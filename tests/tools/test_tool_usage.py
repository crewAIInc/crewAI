import json
import random
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from crewai import Agent, Task
from crewai.tools import BaseTool
from crewai.tools.tool_usage import ToolUsage
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.tool_usage_events import (
    ToolSelectionErrorEvent,
    ToolValidateInputErrorEvent,
)


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


def test_validate_tool_input_booleans_and_none():
    # Create a ToolUsage instance with mocks
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    # Input with booleans and None
    tool_input = '{"key1": True, "key2": False, "key3": None}'
    expected_arguments = {"key1": True, "key2": False, "key3": None}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_mixed_types():
    # Create a ToolUsage instance with mocks
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    # Input with mixed types
    tool_input = '{"number": 123, "text": "Some text", "flag": True}'
    expected_arguments = {"number": 123, "text": "Some text", "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_single_quotes():
    # Create a ToolUsage instance with mocks
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    # Input with single quotes instead of double quotes
    tool_input = "{'key': 'value', 'flag': True}"
    expected_arguments = {"key": "value", "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_invalid_json_repairable():
    # Create a ToolUsage instance with mocks
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    # Invalid JSON input that can be repaired
    tool_input = '{"key": "value", "list": [1, 2, 3,]}'
    expected_arguments = {"key": "value", "list": [1, 2, 3]}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_with_special_characters():
    # Create a ToolUsage instance with mocks
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    # Input with special characters
    tool_input = '{"message": "Hello, world! \u263a", "valid": True}'
    expected_arguments = {"message": "Hello, world! ☺", "valid": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_none_input():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    arguments = tool_usage._validate_tool_input(None)
    assert arguments == {}


def test_validate_tool_input_valid_json():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"key": "value", "number": 42, "flag": true}'
    expected_arguments = {"key": "value", "number": 42, "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_python_dict():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = "{'key': 'value', 'number': 42, 'flag': True}"
    expected_arguments = {"key": "value", "number": 42, "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_json5_unquoted_keys():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = "{key: 'value', number: 42, flag: true}"
    expected_arguments = {"key": "value", "number": 42, "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_with_trailing_commas():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"key": "value", "number": 42, "flag": true,}'
    expected_arguments = {"key": "value", "number": 42, "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_invalid_input():
    # Create mock agent with proper string values
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"  # Must be a string
    mock_agent.role = "test_agent_role"  # Must be a string
    mock_agent._original_role = "test_agent_role"  # Must be a string
    mock_agent.i18n = MagicMock()
    mock_agent.verbose = False

    # Create mock action with proper string value
    mock_action = MagicMock()
    mock_action.tool = "test_tool"  # Must be a string
    mock_action.tool_input = "test_input"  # Must be a string

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=mock_agent,
        action=mock_action,
    )

    invalid_inputs = [
        "Just a string",
        "['list', 'of', 'values']",
        "12345",
        "",
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception) as e_info:
            tool_usage._validate_tool_input(invalid_input)
        assert (
            "Tool input must be a valid dictionary in JSON or Python literal format"
            in str(e_info.value)
        )

    # Test for None input separately
    arguments = tool_usage._validate_tool_input(None)
    assert arguments == {}


def test_validate_tool_input_complex_structure():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = """
    {
        "user": {
            "name": "Alice",
            "age": 30
        },
        "items": [
            {"id": 1, "value": "Item1"},
            {"id": 2, "value": "Item2",}
        ],
        "active": true,
    }
    """
    expected_arguments = {
        "user": {"name": "Alice", "age": 30},
        "items": [
            {"id": 1, "value": "Item1"},
            {"id": 2, "value": "Item2"},
        ],
        "active": True,
    }

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_code_content():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"filename": "script.py", "content": "def hello():\\n    print(\'Hello, world!\')"}'
    expected_arguments = {
        "filename": "script.py",
        "content": "def hello():\n    print('Hello, world!')",
    }

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_with_escaped_quotes():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"text": "He said, \\"Hello, world!\\""}'
    expected_arguments = {"text": 'He said, "Hello, world!"'}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_large_json_content():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

    # Simulate a large JSON content
    tool_input = (
        '{"data": ' + json.dumps([{"id": i, "value": i * 2} for i in range(1000)]) + "}"
    )
    expected_arguments = {"data": [{"id": i, "value": i * 2} for i in range(1000)]}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_tool_selection_error_event_direct():
    """Test tool selection error event emission directly from ToolUsage class."""
    mock_agent = MagicMock()
    mock_agent.key = "test_key"
    mock_agent.role = "test_role"
    mock_agent.i18n = MagicMock()
    mock_agent.verbose = False

    mock_task = MagicMock()
    mock_tools_handler = MagicMock()

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool"

        def _run(self, input: dict) -> str:
            return "test result"

    test_tool = TestTool()

    tool_usage = ToolUsage(
        tools_handler=mock_tools_handler,
        tools=[test_tool],
        original_tools=[test_tool],
        tools_description="Test Tool Description",
        tools_names="Test Tool",
        task=mock_task,
        function_calling_llm=None,
        agent=mock_agent,
        action=MagicMock(),
    )

    received_events = []

    @crewai_event_bus.on(ToolSelectionErrorEvent)
    def event_handler(source, event):
        received_events.append(event)

    with pytest.raises(Exception) as exc_info:
        tool_usage._select_tool("Non Existent Tool")
    assert len(received_events) == 1
    event = received_events[0]
    assert isinstance(event, ToolSelectionErrorEvent)
    assert event.agent_key == "test_key"
    assert event.agent_role == "test_role"
    assert event.tool_name == "Non Existent Tool"
    assert event.tool_args == {}
    assert event.tool_class == "Test Tool Description"
    assert "don't exist" in event.error

    received_events.clear()
    with pytest.raises(Exception) as exc_info:
        tool_usage._select_tool("")

    assert len(received_events) == 1
    event = received_events[0]
    assert isinstance(event, ToolSelectionErrorEvent)
    assert event.agent_key == "test_key"
    assert event.agent_role == "test_role"
    assert event.tool_name == ""
    assert event.tool_args == {}
    assert event.tool_class == "Test Tool Description"
    assert "forgot the Action name" in event.error


def test_tool_validate_input_error_event():
    """Test tool validation input error event emission from ToolUsage class."""
    # Mock agent and required components
    mock_agent = MagicMock()
    mock_agent.key = "test_key"
    mock_agent.role = "test_role"
    mock_agent.verbose = False
    mock_agent._original_role = "test_role"

    # Mock i18n with error message
    mock_i18n = MagicMock()
    mock_i18n.errors.return_value = (
        "Tool input must be a valid dictionary in JSON or Python literal format"
    )
    mock_agent.i18n = mock_i18n

    # Mock task and tools handler
    mock_task = MagicMock()
    mock_tools_handler = MagicMock()

    # Mock printer
    mock_printer = MagicMock()

    # Create test tool
    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool"

        def _run(self, input: dict) -> str:
            return "test result"

    test_tool = TestTool()

    # Create ToolUsage instance
    tool_usage = ToolUsage(
        tools_handler=mock_tools_handler,
        tools=[test_tool],
        original_tools=[test_tool],
        tools_description="Test Tool Description",
        tools_names="Test Tool",
        task=mock_task,
        function_calling_llm=None,
        agent=mock_agent,
        action=MagicMock(tool="test_tool"),
    )
    tool_usage._printer = mock_printer

    # Mock all parsing attempts to fail
    with (
        patch("json.loads", side_effect=json.JSONDecodeError("Test Error", "", 0)),
        patch("ast.literal_eval", side_effect=ValueError),
        patch("json5.loads", side_effect=json.JSONDecodeError("Test Error", "", 0)),
        patch("json_repair.repair_json", side_effect=Exception("Failed to repair")),
    ):
        received_events = []

        @crewai_event_bus.on(ToolValidateInputErrorEvent)
        def event_handler(source, event):
            received_events.append(event)

        # Test invalid input
        invalid_input = "invalid json {[}"
        with pytest.raises(Exception) as exc_info:
            tool_usage._validate_tool_input(invalid_input)

        # Verify event was emitted
        assert len(received_events) == 1, "Expected one event to be emitted"
        event = received_events[0]
        assert isinstance(event, ToolValidateInputErrorEvent)
        assert event.agent_key == "test_key"
        assert event.agent_role == "test_role"
        assert event.tool_name == "test_tool"
        assert "must be a valid dictionary" in event.error
