"""Tests for tool call streaming events across LLM providers.

These tests verify that when streaming is enabled and the LLM makes a tool call,
the stream chunk events include proper tool call information with
call_type=LLMCallType.TOOL_CALL.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.events.types.llm_events import LLMCallType, LLMStreamChunkEvent, ToolCall
from crewai.llm import LLM


@pytest.fixture
def get_temperature_tool_schema() -> dict[str, Any]:
    """Create a temperature tool schema for native function calling."""
    return {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get the current temperature in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get the temperature for.",
                    }
                },
                "required": ["city"],
            },
        },
    }


@pytest.fixture
def mock_emit() -> MagicMock:
    """Mock the event bus emit function."""
    from crewai.events.event_bus import CrewAIEventsBus

    with patch.object(CrewAIEventsBus, "emit") as mock:
        yield mock


def get_tool_call_events(mock_emit: MagicMock) -> list[LLMStreamChunkEvent]:
    """Extract tool call streaming events from mock emit calls."""
    tool_call_events = []
    for call in mock_emit.call_args_list:
        event = call[1].get("event") if len(call) > 1 else None
        if isinstance(event, LLMStreamChunkEvent) and event.call_type == LLMCallType.TOOL_CALL:
            tool_call_events.append(event)
    return tool_call_events


def get_all_stream_events(mock_emit: MagicMock) -> list[LLMStreamChunkEvent]:
    """Extract all streaming events from mock emit calls."""
    stream_events = []
    for call in mock_emit.call_args_list:
        event = call[1].get("event") if len(call) > 1 else None
        if isinstance(event, LLMStreamChunkEvent):
            stream_events.append(event)
    return stream_events


class TestOpenAIToolCallStreaming:
    """Tests for OpenAI provider tool call streaming events."""

    @pytest.mark.vcr()
    def test_openai_streaming_emits_tool_call_events(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that OpenAI streaming emits tool call events with correct call_type."""
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in San Francisco?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        tool_call_events = get_tool_call_events(mock_emit)

        assert len(tool_call_events) > 0, "Should receive tool call streaming events"

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None
        assert isinstance(first_tool_call_event.tool_call, ToolCall)
        assert first_tool_call_event.tool_call.function is not None
        assert first_tool_call_event.tool_call.function.name == "get_current_temperature"
        assert first_tool_call_event.tool_call.type == "function"
        assert first_tool_call_event.tool_call.index >= 0


class TestToolCallStreamingEventStructure:
    """Tests for the structure and content of tool call streaming events."""

    @pytest.mark.vcr()
    def test_tool_call_event_accumulates_arguments(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that tool call events accumulate arguments progressively."""
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in San Francisco?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        tool_call_events = get_tool_call_events(mock_emit)

        assert len(tool_call_events) >= 2, "Should receive multiple tool call streaming events"

        for evt in tool_call_events:
            assert evt.tool_call is not None
            assert evt.tool_call.function is not None

    @pytest.mark.vcr()
    def test_tool_call_events_have_consistent_tool_id(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that all events for the same tool call have the same tool ID."""
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in San Francisco?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        tool_call_events = get_tool_call_events(mock_emit)

        assert len(tool_call_events) >= 1, "Should receive tool call streaming events"

        if len(tool_call_events) > 1:
            events_by_index: dict[int, list[LLMStreamChunkEvent]] = {}
            for evt in tool_call_events:
                if evt.tool_call is not None:
                    idx = evt.tool_call.index
                    if idx not in events_by_index:
                        events_by_index[idx] = []
                    events_by_index[idx].append(evt)

            for idx, evts in events_by_index.items():
                ids = [
                    e.tool_call.id
                    for e in evts
                    if e.tool_call is not None and e.tool_call.id
                ]
                if ids:
                    assert len(set(ids)) == 1, f"Tool call ID should be consistent for index {idx}"


class TestMixedStreamingEvents:
    """Tests for scenarios with both text and tool call streaming events."""

    @pytest.mark.vcr()
    def test_streaming_distinguishes_text_and_tool_calls(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that streaming correctly distinguishes between text chunks and tool calls."""
        llm = LLM(model="openai/gpt-4o-mini", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in San Francisco?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        all_events = get_all_stream_events(mock_emit)
        tool_call_events = get_tool_call_events(mock_emit)

        assert len(all_events) >= 1, "Should receive streaming events"

        for event in tool_call_events:
            assert event.call_type == LLMCallType.TOOL_CALL
            assert event.tool_call is not None


class TestGeminiToolCallStreaming:
    """Tests for Gemini provider tool call streaming events."""

    @pytest.mark.vcr()
    def test_gemini_streaming_emits_tool_call_events(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that Gemini streaming emits tool call events with correct call_type."""
        llm = LLM(model="gemini/gemini-2.0-flash", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in San Francisco?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        tool_call_events = get_tool_call_events(mock_emit)

        assert len(tool_call_events) > 0, "Should receive tool call streaming events"

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None
        assert isinstance(first_tool_call_event.tool_call, ToolCall)
        assert first_tool_call_event.tool_call.function is not None
        assert first_tool_call_event.tool_call.function.name == "get_current_temperature"
        assert first_tool_call_event.tool_call.type == "function"

    @pytest.mark.vcr()
    def test_gemini_streaming_multiple_tool_calls_unique_ids(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that Gemini streaming assigns unique IDs to multiple tool calls."""
        llm = LLM(model="gemini/gemini-2.0-flash", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in Paris and London?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        tool_call_events = get_tool_call_events(mock_emit)

        assert len(tool_call_events) >= 2, "Should receive at least 2 tool call events"

        tool_ids = [
            evt.tool_call.id
            for evt in tool_call_events
            if evt.tool_call is not None and evt.tool_call.id
        ]
        assert len(set(tool_ids)) >= 2, "Each tool call should have a unique ID"


class TestAzureToolCallStreaming:
    """Tests for Azure provider tool call streaming events."""

    @pytest.mark.vcr()
    def test_azure_streaming_emits_tool_call_events(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that Azure streaming emits tool call events with correct call_type."""
        llm = LLM(model="azure/gpt-4o-mini", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in San Francisco?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        tool_call_events = get_tool_call_events(mock_emit)

        assert len(tool_call_events) > 0, "Should receive tool call streaming events"

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None
        assert isinstance(first_tool_call_event.tool_call, ToolCall)
        assert first_tool_call_event.tool_call.function is not None
        assert first_tool_call_event.tool_call.function.name == "get_current_temperature"
        assert first_tool_call_event.tool_call.type == "function"


class TestAnthropicToolCallStreaming:
    """Tests for Anthropic provider tool call streaming events."""

    @pytest.mark.vcr()
    def test_anthropic_streaming_emits_tool_call_events(
        self, get_temperature_tool_schema: dict[str, Any], mock_emit: MagicMock
    ) -> None:
        """Test that Anthropic streaming emits tool call events with correct call_type."""
        llm = LLM(model="anthropic/claude-3-5-haiku-latest", stream=True)

        llm.call(
            messages=[
                {"role": "user", "content": "What is the temperature in San Francisco?"},
            ],
            tools=[get_temperature_tool_schema],
            available_functions={
                "get_current_temperature": lambda city: f"The temperature in {city} is 72°F"
            },
        )

        tool_call_events = get_tool_call_events(mock_emit)

        assert len(tool_call_events) > 0, "Should receive tool call streaming events"

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None
        assert isinstance(first_tool_call_event.tool_call, ToolCall)
        assert first_tool_call_event.tool_call.function is not None
        assert first_tool_call_event.tool_call.function.name == "get_current_temperature"
        assert first_tool_call_event.tool_call.type == "function"