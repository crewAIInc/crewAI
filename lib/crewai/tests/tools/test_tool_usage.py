import datetime
from collections.abc import Callable
import json
import random
import threading
import time
from unittest.mock import MagicMock, patch

from crewai import Agent, Task
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.parser import AgentAction
from crewai.agents.tools_handler import ToolsHandler
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import (
    ToolSelectionErrorEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    ToolValidateInputErrorEvent,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    clear_after_tool_call_hooks,
    register_after_tool_call_hook,
)
from crewai.tools import BaseTool
from crewai.tools.tool_calling import ToolCalling
from crewai.tools.tool_usage import ToolUsage, ToolUsageError
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from pydantic import BaseModel, Field
import pytest


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
        return random.randint(min_value, max_value)  # noqa: S311


class SearchOutput(BaseModel):
    query: str
    score: float


class TypedSearchTool(BaseTool):
    name: str = "typed_search"
    description: str = "Search for a query"

    def _run(self, query: str) -> SearchOutput:
        return SearchOutput(query=query, score=0.7)


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
        tool._run(10, 1)


def test_random_number_tool_schema():
    tool = RandomNumberTool()

    schema = tool.args_schema.model_json_schema()

    schema_str = json.dumps(schema)

    assert "min_value" in schema_str
    assert "max_value" in schema_str

    schema_dict = json.loads(schema_str)

    assert schema_dict["properties"]["min_value"]["type"] == "integer"
    assert schema_dict["properties"]["max_value"]["type"] == "integer"

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
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    rendered = tool_usage._render()

    assert "Tool Name: random_number_generator" in rendered
    assert "Tool Arguments:" in rendered
    assert (
        "Tool Description: Generates a random number within a specified range"
        in rendered
    )

    assert '"min_value"' in rendered
    assert '"max_value"' in rendered
    assert '"type": "integer"' in rendered
    assert '"description": "The minimum value of the range (inclusive)"' in rendered
    assert '"description": "The maximum value of the range (inclusive)"' in rendered


def test_tool_usage_returns_json_agent_text_for_typed_output():
    tool = TypedSearchTool().to_structured_tool()
    tool_usage = ToolUsage(
        tools_handler=None,
        tools=[tool],
        task=None,
        function_calling_llm=MagicMock(),
        agent=None,
        action=MagicMock(),
    )

    result = tool_usage.use(
        calling=ToolCalling(
            tool_name="typed_search",
            arguments={"query": "crew"},
        ),
        tool_string='Action: typed_search\nAction Input: {"query": "crew"}',
    )

    assert json.loads(result) == {"query": "crew", "score": 0.7}


def test_tool_usage_cache_callback_receives_raw_typed_output():
    raw_results: list[object] = []

    def cache_result(_args: object, result: object) -> bool:
        raw_results.append(result)
        return True

    class CacheAwareTypedSearchTool(TypedSearchTool):
        cache_function: Callable = cache_result

    tools_handler = MagicMock()
    tools_handler.cache = None
    tools_handler.last_used_tool = None
    tool = CacheAwareTypedSearchTool().to_structured_tool()
    tool_usage = ToolUsage(
        tools_handler=tools_handler,
        tools=[tool],
        task=None,
        function_calling_llm=MagicMock(),
        agent=None,
        action=MagicMock(),
    )

    result = tool_usage.use(
        calling=ToolCalling(
            tool_name="typed_search",
            arguments={"query": "crew"},
        ),
        tool_string='Action: typed_search\nAction Input: {"query": "crew"}',
    )

    assert json.loads(result) == {"query": "crew", "score": 0.7}
    assert raw_results == [SearchOutput(query="crew", score=0.7)]
    tools_handler.on_tool_use.assert_called_once()
    assert tools_handler.on_tool_use.call_args.kwargs["output"] == SearchOutput(
        query="crew",
        score=0.7,
    )


def test_react_tool_hooks_receive_agent_text_and_raw_cached_typed_output():
    structured_tool = TypedSearchTool().to_structured_tool()
    tools_handler = ToolsHandler(cache=CacheHandler())
    seen_results: list[tuple[str | None, object]] = []

    def after_hook(context: ToolCallHookContext) -> None:
        seen_results.append((context.tool_result, context.raw_tool_result))

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)

    action = AgentAction(
        thought="",
        tool="typed_search",
        tool_input='{"query": "crew"}',
        text='Action: typed_search\nAction Input: {"query": "crew"}',
    )

    try:
        first = execute_tool_and_check_finality(
            agent_action=action,
            tools=[structured_tool],
            tools_handler=tools_handler,
        )
        tools_handler.last_used_tool = None
        second = execute_tool_and_check_finality(
            agent_action=action,
            tools=[structured_tool],
            tools_handler=tools_handler,
        )
    finally:
        clear_after_tool_call_hooks()

    assert json.loads(first.result) == {"query": "crew", "score": 0.7}
    assert json.loads(second.result) == {"query": "crew", "score": 0.7}
    assert seen_results == [
        ('{"query":"crew","score":0.7}', SearchOutput(query="crew", score=0.7)),
        ('{"query":"crew","score":0.7}', SearchOutput(query="crew", score=0.7)),
    ]


def test_last_raw_result_falls_back_only_until_recorded():
    tool_usage = ToolUsage(
        tools_handler=None,
        tools=[],
        task=None,
        function_calling_llm=MagicMock(),
        agent=None,
        action=MagicMock(),
    )

    assert tool_usage.get_last_raw_result("formatted result") == "formatted result"

    tool_usage.last_raw_result = None

    assert tool_usage.get_last_raw_result("formatted result") is None


def test_validate_tool_input_booleans_and_none():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"key1": True, "key2": False, "key3": None}'
    expected_arguments = {"key1": True, "key2": False, "key3": None}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_mixed_types():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"number": 123, "text": "Some text", "flag": True}'
    expected_arguments = {"number": 123, "text": "Some text", "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_single_quotes():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = "{'key': 'value', 'flag': True}"
    expected_arguments = {"key": "value", "flag": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_invalid_json_repairable():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"key": "value", "list": [1, 2, 3,]}'
    expected_arguments = {"key": "value", "list": [1, 2, 3]}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_validate_tool_input_with_special_characters():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    tool_input = '{"message": "Hello, world! \u263a", "valid": True}'
    expected_arguments = {"message": "Hello, world! ☺", "valid": True}

    arguments = tool_usage._validate_tool_input(tool_input)
    assert arguments == expected_arguments


def test_original_tool_calling_non_dict_arguments_raises_tool_usage_error():
    """Regression test for #6430.

    The non-dict arguments branch in `_original_tool_calling` used a bare
    `raise` with no active exception, so hitting it with `raise_error=True`
    crashed with `RuntimeError: No active exception to re-raise` instead of
    raising a meaningful `ToolUsageError`.
    """
    tool = RandomNumberTool()
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[tool],
        task=MagicMock(),
        function_calling_llm=MagicMock(),
        agent=MagicMock(),
        action=MagicMock(),
    )

    with (
        patch.object(
            tool_usage, "_validate_tool_input", return_value=["not", "a", "dict"]
        ),
        patch.object(tool_usage, "_select_tool", return_value=tool),
    ):
        with pytest.raises(ToolUsageError):
            tool_usage._original_tool_calling("", raise_error=True)

        result = tool_usage._original_tool_calling("")
        assert isinstance(result, ToolUsageError)


def test_validate_tool_input_none_input():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
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
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.verbose = False

    mock_action = MagicMock()
    mock_action.tool = "test_tool"
    mock_action.tool_input = "test_input"

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
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

    arguments = tool_usage._validate_tool_input(None)
    assert arguments == {}


def test_validate_tool_input_complex_structure():
    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[],
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
        task=MagicMock(),
        function_calling_llm=None,
        agent=MagicMock(),
        action=MagicMock(),
    )

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
        task=mock_task,
        function_calling_llm=None,
        agent=mock_agent,
        action=MagicMock(),
    )

    received_events = []
    first_event_received = threading.Event()
    second_event_received = threading.Event()

    @crewai_event_bus.on(ToolSelectionErrorEvent)
    def event_handler(source, event):
        received_events.append(event)
        if event.tool_name == "Non Existent Tool":
            first_event_received.set()
        elif event.tool_name == "":
            second_event_received.set()

    with pytest.raises(Exception):  # noqa: B017
        tool_usage._select_tool("Non Existent Tool")

    assert first_event_received.wait(timeout=5), "Timeout waiting for first event"
    assert len(received_events) == 1
    event = received_events[0]
    assert isinstance(event, ToolSelectionErrorEvent)
    assert event.agent_key == "test_key"
    assert event.agent_role == "test_role"
    assert event.tool_name == "Non Existent Tool"
    assert event.tool_args == {}
    assert "Tool Name: test_tool" in event.tool_class
    assert "A test tool" in event.tool_class
    assert "don't exist" in event.error

    with pytest.raises(Exception):  # noqa: B017
        tool_usage._select_tool("")

    assert second_event_received.wait(timeout=5), "Timeout waiting for second event"
    assert len(received_events) == 2
    event = received_events[1]
    assert isinstance(event, ToolSelectionErrorEvent)
    assert event.agent_key == "test_key"
    assert event.agent_role == "test_role"
    assert event.tool_name == ""
    assert event.tool_args == {}
    assert "test_tool" in event.tool_class
    assert "forgot the Action name" in event.error


def test_tool_validate_input_error_event():
    """Test tool validation input error event emission from ToolUsage class."""
    mock_agent = MagicMock()
    mock_agent.key = "test_key"
    mock_agent.role = "test_role"
    mock_agent.verbose = False
    mock_agent._original_role = "test_role"

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
        task=mock_task,
        function_calling_llm=None,
        agent=mock_agent,
        action=MagicMock(tool="test_tool"),
    )
    with (
        patch("json.loads", side_effect=json.JSONDecodeError("Test Error", "", 0)),
        patch("ast.literal_eval", side_effect=ValueError),
        patch("json5.loads", side_effect=json.JSONDecodeError("Test Error", "", 0)),
        patch("json_repair.repair_json", side_effect=Exception("Failed to repair")),
    ):
        received_events = []
        condition = threading.Condition()

        @crewai_event_bus.on(ToolValidateInputErrorEvent)
        def event_handler(source, event):
            with condition:
                received_events.append(event)
                condition.notify()

        invalid_input = "invalid json {[}"
        with pytest.raises(Exception):  # noqa: B017
            tool_usage._validate_tool_input(invalid_input)

        with condition:
            if not received_events:
                condition.wait(timeout=5)

        assert len(received_events) == 1, "Expected one event to be emitted"
        event = received_events[0]
        assert isinstance(event, ToolValidateInputErrorEvent)
        assert event.agent_key == "test_key"
        assert event.agent_role == "test_role"
        assert event.tool_name == "test_tool"
        assert "must be a valid dictionary" in event.error


def test_tool_usage_finished_event_with_result():
    """Test that ToolUsageFinishedEvent is emitted with correct result attributes."""
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.verbose = False

    mock_task = MagicMock()
    mock_task.delegations = 0
    mock_task.name = "Test Task"
    mock_task.description = "A test task for tool usage"
    mock_task.id = "test-task-id"

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool"

        def _run(self, input: dict) -> str:
            return "test result"

    test_tool = TestTool()

    mock_tool_calling = MagicMock()
    mock_tool_calling.arguments = {"arg1": "value1"}

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[test_tool],
        task=mock_task,
        function_calling_llm=None,
        agent=mock_agent,
        action=MagicMock(),
    )

    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def event_handler(source, event):
        received_events.append(event)
        event_received.set()

    started_at = time.time()
    result = "test output result"
    tool_usage.on_tool_use_finished(
        tool=test_tool,
        tool_calling=mock_tool_calling,
        from_cache=False,
        started_at=started_at,
        result=result,
    )

    assert event_received.wait(timeout=5), "Timeout waiting for event"
    assert len(received_events) == 1, "Expected one event to be emitted"
    event = received_events[0]
    assert isinstance(event, ToolUsageFinishedEvent)

    assert event.agent_key == "test_agent_key"
    assert event.agent_role == "test_agent_role"
    assert event.tool_name == "test_tool"
    assert event.tool_args == {"arg1": "value1"}
    assert event.tool_class == "TestTool"
    assert event.run_attempts == 1
    assert event.delegations == 0
    assert event.from_cache is False
    assert event.output == "test output result"
    assert isinstance(event.started_at, datetime.datetime)
    assert isinstance(event.finished_at, datetime.datetime)
    assert event.type == "tool_usage_finished"


def test_tool_usage_finished_event_with_cached_result():
    """Test that ToolUsageFinishedEvent is emitted with correct result attributes when using cached result."""
    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.verbose = False

    mock_task = MagicMock()
    mock_task.delegations = 0
    mock_task.name = "Test Task"
    mock_task.description = "A test task for tool usage"
    mock_task.id = "test-task-id"

    class TestTool(BaseTool):
        name: str = "Test Tool"
        description: str = "A test tool"

        def _run(self, input: dict) -> str:
            return "test result"

    test_tool = TestTool()

    mock_tool_calling = MagicMock()
    mock_tool_calling.arguments = {"arg1": "value1"}

    tool_usage = ToolUsage(
        tools_handler=MagicMock(),
        tools=[test_tool],
        task=mock_task,
        function_calling_llm=None,
        agent=mock_agent,
        action=MagicMock(),
    )

    received_events = []
    event_received = threading.Event()

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def event_handler(source, event):
        received_events.append(event)
        event_received.set()

    started_at = time.time()
    result = "cached test output result"
    tool_usage.on_tool_use_finished(
        tool=test_tool,
        tool_calling=mock_tool_calling,
        from_cache=True,
        started_at=started_at,
        result=result,
    )

    assert event_received.wait(timeout=5), "Timeout waiting for event"
    assert len(received_events) == 1, "Expected one event to be emitted"
    event = received_events[0]
    assert isinstance(event, ToolUsageFinishedEvent)

    assert event.agent_key == "test_agent_key"
    assert event.agent_role == "test_agent_role"
    assert event.tool_name == "test_tool"
    assert event.tool_args == {"arg1": "value1"}
    assert event.tool_class == "TestTool"
    assert event.run_attempts == 1
    assert event.delegations == 0
    assert event.from_cache is True
    assert event.output == "cached test output result"
    assert isinstance(event.started_at, datetime.datetime)
    assert isinstance(event.finished_at, datetime.datetime)
    assert event.type == "tool_usage_finished"


def test_tool_error_does_not_emit_finished_event():
    from crewai.tools.tool_calling import ToolCalling

    class FailingTool(BaseTool):
        name: str = "Failing Tool"
        description: str = "A tool that always fails"

        def _run(self, **kwargs) -> str:
            raise ValueError("Intentional failure")

    failing_tool = FailingTool().to_structured_tool()

    mock_agent = MagicMock()
    mock_agent.key = "test_agent_key"
    mock_agent.role = "test_agent_role"
    mock_agent._original_role = "test_agent_role"
    mock_agent.verbose = False
    mock_agent.fingerprint = None

    mock_task = MagicMock()
    mock_task.delegations = 0
    mock_task.name = "Test Task"
    mock_task.description = "A test task"
    mock_task.id = "test-task-id"

    mock_action = MagicMock()
    mock_action.tool = "failing_tool"
    mock_action.tool_input = "{}"

    tool_usage = ToolUsage(
        tools_handler=MagicMock(cache=None, last_used_tool=None),
        tools=[failing_tool],
        task=mock_task,
        function_calling_llm=None,
        agent=mock_agent,
        action=mock_action,
    )

    started_events = []
    error_events = []
    finished_events = []
    error_received = threading.Event()

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def on_started(source, event):
        if event.tool_name == "failing_tool":
            started_events.append(event)

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def on_error(source, event):
        if event.tool_name == "failing_tool":
            error_events.append(event)
            error_received.set()

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def on_finished(source, event):
        if event.tool_name == "failing_tool":
            finished_events.append(event)

    tool_calling = ToolCalling(tool_name="failing_tool", arguments={})
    tool_usage.use(calling=tool_calling, tool_string="Action: failing_tool")

    assert error_received.wait(timeout=5), "Timeout waiting for error event"
    crewai_event_bus.flush()

    assert len(started_events) >= 1, "Expected at least one ToolUsageStartedEvent"
    assert len(error_events) >= 1, "Expected at least one ToolUsageErrorEvent"
    assert len(finished_events) == 0, (
        "ToolUsageFinishedEvent should NOT be emitted after ToolUsageErrorEvent"
    )
