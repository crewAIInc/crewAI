import asyncio
from collections.abc import Callable
import datetime
import json
import random
import threading
import time
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field
import pytest

from crewai import Agent, Task
from crewai.agent.utils import process_tool_results
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
from crewai.tools.tool_results import active_tool_results, tool_results_collector
from crewai.tools.tool_usage import ToolUsage
from crewai.utilities.tool_utils import (
    aexecute_tool_and_check_finality,
    execute_tool_and_check_finality,
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


class AmountTool(BaseTool):
    name: str = "amount_tool"
    description: str = "Reports an amount."
    result_as_answer: bool = True

    def _run(self) -> str:
        return "amount: 100"

    async def _arun(self) -> str:
        return "amount: 100"


class FailingAmountTool(BaseTool):
    name: str = "amount_tool"
    description: str = "Reports an amount but fails."
    result_as_answer: bool = True

    def _run(self) -> str:
        raise RuntimeError("upstream timeout")

    async def _arun(self) -> str:
        raise RuntimeError("upstream timeout")


def _amount_action() -> AgentAction:
    return AgentAction(
        thought="",
        tool="amount_tool",
        tool_input="{}",
        text="Action: amount_tool\nAction Input: {}",
    )


def _amount_agent() -> Agent:
    agent = Agent(
        role="Teller",
        goal="Report amounts",
        backstory="You report amounts.",
        tools=[AmountTool()],
    )
    agent.tools_results = []
    return agent


def test_react_result_as_answer_keeps_the_after_hook_replacement():
    """result_as_answer must answer with the hook's value, not the raw result.

    ToolUsage records the pre-hook result on the agent, and
    process_tool_results() reads that record back to honour result_as_answer.
    The executor's own answer already carries the replacement, so the two have
    to agree -- otherwise the replacement is applied and then overwritten.
    """
    agent = _amount_agent()

    def after_hook(context: ToolCallHookContext) -> str:
        return "Amount: 100 USD"

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)
    try:
        tool_result = execute_tool_and_check_finality(
            agent_action=_amount_action(),
            tools=[AmountTool().to_structured_tool()],
            tools_handler=ToolsHandler(cache=CacheHandler()),
            agent=agent,
        )
    finally:
        clear_after_tool_call_hooks()

    assert tool_result.result == "Amount: 100 USD"
    assert tool_result.result_as_answer is True
    assert process_tool_results(agent, tool_result.result) == "Amount: 100 USD"


@pytest.mark.asyncio
async def test_async_react_result_as_answer_keeps_the_after_hook_replacement():
    """The async ReAct path mirrors the sync one."""
    agent = _amount_agent()

    def after_hook(context: ToolCallHookContext) -> str:
        return "Amount: 100 USD"

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)
    try:
        tool_result = await aexecute_tool_and_check_finality(
            agent_action=_amount_action(),
            tools=[AmountTool().to_structured_tool()],
            tools_handler=ToolsHandler(cache=CacheHandler()),
            agent=agent,
        )
    finally:
        clear_after_tool_call_hooks()

    assert tool_result.result == "Amount: 100 USD"
    assert process_tool_results(agent, tool_result.result) == "Amount: 100 USD"


def test_react_result_as_answer_unchanged_without_an_after_hook():
    """Control: with no hook registered the raw tool result is still the answer."""
    agent = _amount_agent()
    clear_after_tool_call_hooks()
    tool_result = execute_tool_and_check_finality(
        agent_action=_amount_action(),
        tools=[AmountTool().to_structured_tool()],
        tools_handler=ToolsHandler(cache=CacheHandler()),
        agent=agent,
    )

    assert tool_result.result == "amount: 100"
    assert process_tool_results(agent, tool_result.result) == "amount: 100"


def test_react_after_hook_that_returns_none_leaves_the_result_alone():
    """Control: a monitoring hook that returns None must not blank the answer."""
    agent = _amount_agent()
    seen: list[str | None] = []

    def after_hook(context: ToolCallHookContext) -> None:
        seen.append(context.tool_result)

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)
    try:
        tool_result = execute_tool_and_check_finality(
            agent_action=_amount_action(),
            tools=[AmountTool().to_structured_tool()],
            tools_handler=ToolsHandler(cache=CacheHandler()),
            agent=agent,
        )
    finally:
        clear_after_tool_call_hooks()

    assert seen == ["amount: 100"]
    assert tool_result.result == "amount: 100"
    assert process_tool_results(agent, tool_result.result) == "amount: 100"


def test_react_result_as_answer_failed_call_does_not_overwrite_earlier_result():
    """A subsequent failed call must not overwrite an earlier successful result.

    When a tool call fails, ToolUsage does not append an entry to agent.tools_results.
    The after hook still executes on the error string, but _record_after_hook_result
    must not search backwards past the current invocation boundary (start_index)
    to corrupt the earlier successful call with error text.
    """
    agent = _amount_agent()

    def after_hook(context: ToolCallHookContext) -> str:
        return f"[audited] {context.tool_result}"

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)
    try:
        # Call 1: Success
        res1 = execute_tool_and_check_finality(
            agent_action=_amount_action(),
            tools=[AmountTool().to_structured_tool()],
            tools_handler=ToolsHandler(cache=CacheHandler()),
            agent=agent,
        )
        assert res1.result == "[audited] amount: 100"
        assert len(agent.tools_results) == 1
        assert agent.tools_results[0]["result"] == "[audited] amount: 100"

        # Call 2: Failure (raises RuntimeError)
        res2 = execute_tool_and_check_finality(
            agent_action=_amount_action(),
            tools=[FailingAmountTool().to_structured_tool()],
            tools_handler=ToolsHandler(cache=CacheHandler()),
            agent=agent,
        )
        assert "upstream timeout" in res2.result
        # The failed call must not append or corrupt the earlier successful entry
        assert len(agent.tools_results) == 1
        assert agent.tools_results[0]["result"] == "[audited] amount: 100"

        # process_tool_results should still retain the successful answer, NOT the error
        final_answer = process_tool_results(agent, "final thought")
        assert final_answer == "[audited] amount: 100"
        assert "upstream timeout" not in final_answer
    finally:
        clear_after_tool_call_hooks()


def test_agent_tools_results_cleared_between_tasks():
    """agent.tools_results must be cleared before preparing a new task.

    Ensures that result_as_answer tools from task 1 do not leak into task 2.
    """
    agent = _amount_agent()
    # Simulate task 1 having populated tools_results
    agent.tools_results = [
        {"tool_name": "amount_tool", "result": "Amount: 100 USD", "result_as_answer": True}
    ]

    task2 = Task(
        description="Write a summary report",
        expected_output="Summary report",
        agent=agent,
    )
    # When preparing task 2, tools_results should be reset
    agent._prepare_task_execution(task2, context=None)
    assert agent.tools_results == []


@pytest.mark.asyncio
async def test_async_react_result_as_answer_failed_call_does_not_overwrite_earlier_result():
    """Async twin: A subsequent failed call must not overwrite an earlier successful result."""
    agent = _amount_agent()

    def after_hook(context: ToolCallHookContext) -> str:
        return f"[audited] {context.tool_result}"

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)
    try:
        # Call 1: Success
        res1 = await aexecute_tool_and_check_finality(
            agent_action=_amount_action(),
            tools=[AmountTool().to_structured_tool()],
            tools_handler=ToolsHandler(cache=CacheHandler()),
            agent=agent,
        )
        assert res1.result == "[audited] amount: 100"
        assert len(agent.tools_results) == 1
        assert agent.tools_results[0]["result"] == "[audited] amount: 100"

        # Call 2: Failure (raises RuntimeError)
        res2 = await aexecute_tool_and_check_finality(
            agent_action=_amount_action(),
            tools=[FailingAmountTool().to_structured_tool()],
            tools_handler=ToolsHandler(cache=CacheHandler()),
            agent=agent,
        )
        assert "upstream timeout" in res2.result
        assert len(agent.tools_results) == 1
        assert agent.tools_results[0]["result"] == "[audited] amount: 100"

        final_answer = process_tool_results(agent, "final thought")
        assert final_answer == "[audited] amount: 100"
        assert "upstream timeout" not in final_answer
    finally:
        clear_after_tool_call_hooks()


def test_react_result_as_answer_warm_cache_hit_preserves_hook_replacement():
    """A warm cache hit must record its result when result_as_answer=True.

    If an earlier call in the same task ran fresh (e.g. Call A returning 100),
    and a later call is served from cache (Call B returning 200), the cached call's
    result must be recorded so process_tool_results returns Call B instead of
    falling back to Call A.
    """
    agent = _amount_agent()
    cache_handler = CacheHandler()
    tools_handler = ToolsHandler(cache=cache_handler)

    def after_hook(context: ToolCallHookContext) -> str:
        return f"[formatted] {context.tool_result}"

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)
    try:
        # Pre-seed cache for input {"code": "cached"}
        cache_handler.add(
            tool="amount_tool",
            input='{"code": "cached"}',
            output="amount: 200",
        )

        # Call A: fresh call with {"code": "fresh"}
        action_a = AgentAction(
            thought="",
            tool="amount_tool",
            tool_input='{"code": "fresh"}',
            text='Action: amount_tool\nAction Input: {"code": "fresh"}',
        )
        res_a = execute_tool_and_check_finality(
            agent_action=action_a,
            tools=[AmountTool().to_structured_tool()],
            tools_handler=tools_handler,
            agent=agent,
        )
        assert res_a.result == "[formatted] amount: 100"
        assert len(agent.tools_results) == 1
        assert agent.tools_results[0]["result"] == "[formatted] amount: 100"

        # Call B: cached call with {"code": "cached"}
        action_b = AgentAction(
            thought="",
            tool="amount_tool",
            tool_input='{"code": "cached"}',
            text='Action: amount_tool\nAction Input: {"code": "cached"}',
        )
        res_b = execute_tool_and_check_finality(
            agent_action=action_b,
            tools=[AmountTool().to_structured_tool()],
            tools_handler=tools_handler,
            agent=agent,
        )
        assert res_b.result == "[formatted] amount: 200"
        # Must have recorded the cached result as the latest entry
        assert len(agent.tools_results) == 2
        assert agent.tools_results[-1]["result"] == "[formatted] amount: 200"

        # process_tool_results must honour Call B, NOT Call A
        final_answer = process_tool_results(agent, res_b.result)
        assert final_answer == "[formatted] amount: 200"
    finally:
        clear_after_tool_call_hooks()


@pytest.mark.asyncio
async def test_async_react_result_as_answer_warm_cache_hit_preserves_hook_replacement():
    """Async twin: A warm cache hit must record its result when result_as_answer=True."""
    agent = _amount_agent()
    cache_handler = CacheHandler()
    tools_handler = ToolsHandler(cache=cache_handler)

    def after_hook(context: ToolCallHookContext) -> str:
        return f"[formatted] {context.tool_result}"

    clear_after_tool_call_hooks()
    register_after_tool_call_hook(after_hook)
    try:
        # Pre-seed cache for input {"code": "cached"}
        cache_handler.add(
            tool="amount_tool",
            input='{"code": "cached"}',
            output="amount: 200",
        )

        # Call A: fresh call with {"code": "fresh"}
        action_a = AgentAction(
            thought="",
            tool="amount_tool",
            tool_input='{"code": "fresh"}',
            text='Action: amount_tool\nAction Input: {"code": "fresh"}',
        )
        res_a = await aexecute_tool_and_check_finality(
            agent_action=action_a,
            tools=[AmountTool().to_structured_tool()],
            tools_handler=tools_handler,
            agent=agent,
        )
        assert res_a.result == "[formatted] amount: 100"
        assert len(agent.tools_results) == 1

        # Call B: cached call with {"code": "cached"}
        action_b = AgentAction(
            thought="",
            tool="amount_tool",
            tool_input='{"code": "cached"}',
            text='Action: amount_tool\nAction Input: {"code": "cached"}',
        )
        res_b = await aexecute_tool_and_check_finality(
            agent_action=action_b,
            tools=[AmountTool().to_structured_tool()],
            tools_handler=tools_handler,
            agent=agent,
        )
        assert res_b.result == "[formatted] amount: 200"
        assert len(agent.tools_results) == 2
        assert agent.tools_results[-1]["result"] == "[formatted] amount: 200"

        final_answer = process_tool_results(agent, res_b.result)
        assert final_answer == "[formatted] amount: 200"
    finally:
        clear_after_tool_call_hooks()


def test_agent_sequential_tasks_tool_results_isolation_end_to_end():
    """Sequential tasks on one Agent do not leak result_as_answer across boundaries."""
    agent = _amount_agent()

    # Task 1 uses AmountTool (result_as_answer=True)
    task1 = Task(
        description="Fetch report",
        expected_output="Amount report",
        agent=agent,
    )
    agent._prepare_task_execution(task1, context=None)
    execute_tool_and_check_finality(
        agent_action=_amount_action(),
        tools=[AmountTool().to_structured_tool()],
        tools_handler=ToolsHandler(cache=CacheHandler()),
        agent=agent,
    )
    task1_output = process_tool_results(agent, "agent raw answer")
    assert task1_output == "amount: 100"

    # Task 2 is a pure LLM task with no result_as_answer tools
    task2 = Task(
        description="Write a summary",
        expected_output="Summary",
        agent=agent,
    )
    agent._prepare_task_execution(task2, context=None)
    assert agent.tools_results == []

    # Finalizing task 2 must preserve task 2's own output, NOT task 1's tool output
    task2_output = process_tool_results(agent, "My Task 2 Summary")
    assert task2_output == "My Task 2 Summary"


def test_tool_results_collector_is_execution_scoped():
    """ContextVar collector isolates nested and concurrent execution scopes."""
    assert active_tool_results() is None

    with tool_results_collector() as outer_results:
        assert active_tool_results() is outer_results
        outer_results.append({"tool_name": "outer", "result": 1})

        with tool_results_collector() as inner_results:
            assert active_tool_results() is inner_results
            assert active_tool_results() is not outer_results
            inner_results.append({"tool_name": "inner", "result": 2})
            assert len(active_tool_results()) == 1

        assert active_tool_results() is outer_results
        assert len(active_tool_results()) == 1
        assert outer_results[0]["result"] == 1

    assert active_tool_results() is None


@pytest.mark.asyncio
async def test_agent_concurrent_tasks_tool_results_isolation():
    """Concurrent async tasks sharing an agent must have isolated tool results."""
    async def task_worker(task_id: int):
        with tool_results_collector() as collector:
            for step in range(3):
                active = active_tool_results()
                assert active is collector
                active.append({"task_id": task_id, "step": step})
                await asyncio.sleep(0.01)
            return list(active)

    res1, res2 = await asyncio.gather(task_worker(1), task_worker(2))
    assert all(r["task_id"] == 1 for r in res1)
    assert len(res1) == 3
    assert all(r["task_id"] == 2 for r in res2)
    assert len(res2) == 3


def test_agent_concurrent_threads_tool_results_isolation():
    """Concurrent threads sharing an agent must have isolated tool results."""
    results = {}

    def thread_worker(thread_id: int):
        with tool_results_collector() as collector:
            for step in range(3):
                active = active_tool_results()
                assert active is collector
                active.append({"thread_id": thread_id, "step": step})
                time.sleep(0.01)
            results[thread_id] = list(active)

    t1 = threading.Thread(target=thread_worker, args=(1,))
    t2 = threading.Thread(target=thread_worker, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert all(r["thread_id"] == 1 for r in results[1])
    assert len(results[1]) == 3
    assert all(r["thread_id"] == 2 for r in results[2])
    assert len(results[2]) == 3


@pytest.mark.asyncio
async def test_agent_concurrent_tasks_reset_tools_results_does_not_clear_other_task():
    """Calling reset_tools_results() during a task execution only clears that task's results."""
    agent = _amount_agent()

    async def task_worker(task_id: int, should_reset: bool):
        with tool_results_collector() as collector:
            collector.append({"task_id": task_id, "before": True})
            await asyncio.sleep(0.02)
            if should_reset:
                agent.reset_tools_results()
            await asyncio.sleep(0.02)
            collector.append({"task_id": task_id, "after": True})
            return list(collector)

    res_reset, res_keep = await asyncio.gather(
        task_worker(1, should_reset=True),
        task_worker(2, should_reset=False),
    )

    assert len(res_reset) == 1
    assert res_reset[0] == {"task_id": 1, "after": True}

    assert len(res_keep) == 2
    assert res_keep[0] == {"task_id": 2, "before": True}
    assert res_keep[1] == {"task_id": 2, "after": True}


@pytest.mark.asyncio
async def test_concurrent_tasks_shared_agent_real_react_tool_hooks_isolation():
    """Concurrent tasks on a shared agent exercise real ToolUsage, hooks, and finalization with forced barrier overlap."""
    agent = _amount_agent()

    clear_after_tool_call_hooks()

    def deterministic_after_hook(context: ToolCallHookContext):
        if context.tool_name == "amount_tool":
            code = (
                context.tool_input.get("code")
                if isinstance(context.tool_input, dict)
                else None
            )
            return f"Formatted for {code}: 999"
        return None

    register_after_tool_call_hook(deterministic_after_hook)

    worker1_in_tool = asyncio.Event()
    worker2_in_tool = asyncio.Event()

    class BarrierAmountTool(BaseTool):
        name: str = "amount_tool"
        description: str = "Reports an amount with forced concurrency barrier."
        result_as_answer: bool = True

        def _run(self, code: str = "") -> str:
            return f"raw: {code}"

        async def _arun(self, code: str = "") -> str:
            if code == "task_1":
                worker1_in_tool.set()
                await worker2_in_tool.wait()
            elif code == "task_2":
                worker2_in_tool.set()
                await worker1_in_tool.wait()
            return f"raw: {code}"

    task1 = Task(description="Fetch 1", expected_output="Output 1", agent=agent)
    task2 = Task(description="Fetch 2", expected_output="Output 2", agent=agent)

    action1 = AgentAction(
        thought="",
        tool="amount_tool",
        tool_input='{"code": "task_1"}',
        text='Action: amount_tool\nAction Input: {"code": "task_1"}',
    )
    action2 = AgentAction(
        thought="",
        tool="amount_tool",
        tool_input='{"code": "task_2"}',
        text='Action: amount_tool\nAction Input: {"code": "task_2"}',
    )

    async def worker1():
        with tool_results_collector():
            res = await aexecute_tool_and_check_finality(
                agent_action=action1,
                tools=[BarrierAmountTool().to_structured_tool()],
                tools_handler=ToolsHandler(cache=CacheHandler()),
                agent=agent,
            )
            return agent._finalize_task_execution(task1, res.result)

    async def worker2():
        with tool_results_collector():
            res = await aexecute_tool_and_check_finality(
                agent_action=action2,
                tools=[BarrierAmountTool().to_structured_tool()],
                tools_handler=ToolsHandler(cache=CacheHandler()),
                agent=agent,
            )
            return agent._finalize_task_execution(task2, res.result)

    try:
        res1, res2 = await asyncio.gather(worker1(), worker2())
        assert res1 == "Formatted for task_1: 999"
        assert res2 == "Formatted for task_2: 999"
        assert len(agent.tools_results) == 1
        assert agent.tools_results[0]["result"] in (res1, res2)
    finally:
        clear_after_tool_call_hooks()


def test_prepopulated_agent_tools_results_not_corrupted_by_failed_hook_in_collector():
    """Active collector operations do not mutate pre-populated agent.tools_results."""
    from crewai.utilities.tool_utils import _record_after_hook_result

    agent = _amount_agent()
    original_entry = {
        "tool_name": "amount_tool",
        "result": "original_valid_value",
        "result_as_answer": True,
    }
    agent.tools_results = [original_entry]

    with tool_results_collector() as active:
        _record_after_hook_result(
            agent=agent,
            tool_name="amount_tool",
            result="failed_hook_attempt",
            start_index=len(active),
            result_as_answer=False,
        )
        assert len(active) == 0

    assert len(agent.tools_results) == 1
    assert agent.tools_results[0]["result"] == "original_valid_value"
    assert agent.tools_results[0]["result_as_answer"] is True


def test_nested_agent_execution_preserves_parent_tool_results():
    """Nested execution (such as coworker delegation) isolates child and parent tool results."""
    manager = _amount_agent()
    coworker = _amount_agent()

    # Manager starts a task and accumulates a tool result
    task1 = Task(description="Manager Task", expected_output="Manager Output", agent=manager)
    with tool_results_collector() as manager_results:
        manager._prepare_task_execution(task1, context=None)
        manager_results.append({
            "tool_name": "amount_tool",
            "result": "Manager Report: 100",
            "result_as_answer": True,
        })

        # Coworker executes a delegated task within the same thread/context
        task2 = Task(description="Delegated Task", expected_output="Coworker Output", agent=coworker)
        with tool_results_collector() as coworker_results:
            coworker._prepare_task_execution(task2, context=None)
            assert coworker_results is not manager_results
            coworker_results.append({
                "tool_name": "amount_tool",
                "result": "Coworker Report: 200",
                "result_as_answer": True,
            })
            coworker_output = process_tool_results(coworker, "Delegated summary")
            assert coworker_output == "Coworker Report: 200"

        # Manager resumes; manager's results must be intact and unaffected by coworker
        assert len(manager_results) == 1
        assert manager_results[0]["result"] == "Manager Report: 100"
        manager_output = process_tool_results(manager, "Manager final summary")
        assert manager_output == "Manager Report: 100"

