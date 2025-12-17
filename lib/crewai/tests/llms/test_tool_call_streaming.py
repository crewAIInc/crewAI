"""Tests for tool call streaming events across LLM providers.

These tests verify that when streaming is enabled and the LLM makes a tool call,
the stream chunk events include proper tool call information with
call_type=LLMCallType.TOOL_CALL.
"""

import threading
from typing import Any

import pytest

# ruff: noqa: ARG001

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMCallType, LLMStreamChunkEvent
from crewai.llm import LLM
from crewai.tools import tool


@tool
def get_current_temperature(city: str) -> str:
    """Get the current temperature in a city.

    Args:
        city: The name of the city to get the temperature for.

    Returns:
        A string with the current temperature.
    """
    return f"The current temperature in {city} is 72°F"


@pytest.fixture
def weather_agent() -> Agent:
    """Create a weather agent with the temperature tool for testing."""
    return Agent(
        role="Weather Reporter",
        goal="Report the current weather for cities",
        backstory="You are a weather reporter that provides accurate temperature information.",
        tools=[get_current_temperature],
        allow_delegation=False,
    )


@pytest.fixture
def weather_task(weather_agent: Agent) -> Task:
    """Create a weather task."""
    return Task(
        description="What is the current temperature in San Francisco? Use the get_current_temperature tool.",
        expected_output="The temperature in San Francisco",
        agent=weather_agent,
    )


class TestOpenAIToolCallStreaming:
    """Tests for OpenAI provider tool call streaming events."""

    @pytest.mark.vcr()
    def test_openai_streaming_emits_tool_call_events(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that OpenAI streaming emits tool call events with correct call_type."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        text_chunk_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                elif event.chunk:
                    text_chunk_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="gpt-4o-mini", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        crew.kickoff()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 1,
                timeout=30,
            )

        assert success, "Timeout waiting for tool call streaming events"
        assert len(tool_call_events) > 0, "Should receive tool call streaming events"

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None
        assert first_tool_call_event.tool_call.function is not None
        assert first_tool_call_event.tool_call.function.name == "get_current_temperature"
        assert first_tool_call_event.tool_call.type == "function"
        assert first_tool_call_event.tool_call.index >= 0

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_openai_async_streaming_emits_tool_call_events(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that OpenAI async streaming emits tool call events with correct call_type."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="gpt-4o-mini", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        await crew.kickoff_async()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 1,
                timeout=30,
            )

        assert success, "Timeout waiting for async tool call streaming events"
        assert len(tool_call_events) > 0

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None


class TestAnthropicToolCallStreaming:
    """Tests for Anthropic provider tool call streaming events."""

    @pytest.mark.vcr()
    def test_anthropic_streaming_emits_tool_call_events(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that Anthropic streaming emits tool call events with correct call_type."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="anthropic/claude-sonnet-4-20250514", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        crew.kickoff()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 1,
                timeout=30,
            )

        assert success, "Timeout waiting for Anthropic tool call streaming events"
        assert len(tool_call_events) > 0

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None
        assert first_tool_call_event.tool_call.function is not None
        assert first_tool_call_event.tool_call.function.name == "get_current_temperature"

    @pytest.mark.vcr()
    @pytest.mark.asyncio
    async def test_anthropic_async_streaming_emits_tool_call_events(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that Anthropic async streaming emits tool call events with correct call_type."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="anthropic/claude-sonnet-4-20250514", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        await crew.kickoff_async()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 1,
                timeout=30,
            )

        assert success, "Timeout waiting for Anthropic async tool call streaming events"
        assert len(tool_call_events) > 0
        assert tool_call_events[0].call_type == LLMCallType.TOOL_CALL


class TestGeminiToolCallStreaming:
    """Tests for Gemini provider tool call streaming events."""

    @pytest.mark.vcr()
    def test_gemini_streaming_emits_tool_call_events(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that Gemini streaming emits tool call events with correct call_type."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="gemini/gemini-2.0-flash", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        crew.kickoff()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 1,
                timeout=30,
            )

        assert success, "Timeout waiting for Gemini tool call streaming events"
        assert len(tool_call_events) > 0

        first_tool_call_event = tool_call_events[0]
        assert first_tool_call_event.call_type == LLMCallType.TOOL_CALL
        assert first_tool_call_event.tool_call is not None
        assert first_tool_call_event.tool_call.function is not None
        assert first_tool_call_event.tool_call.function.name == "get_current_temperature"


class TestToolCallStreamingEventStructure:
    """Tests for the structure and content of tool call streaming events."""

    @pytest.mark.vcr()
    def test_tool_call_event_accumulates_arguments(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that tool call events accumulate arguments progressively."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="gpt-4o-mini", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        crew.kickoff()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 2,
                timeout=30,
            )

        assert success, "Timeout waiting for multiple tool call streaming events"

        for evt in tool_call_events:
            assert evt.tool_call is not None
            assert evt.tool_call.function is not None
            assert evt.tool_call.function.arguments is not None

        if len(tool_call_events) > 1:
            args_lengths = [
                len(e.tool_call.function.arguments)
                for e in tool_call_events
                if e.tool_call is not None
            ]
            assert args_lengths[-1] >= args_lengths[0], (
                "Arguments should accumulate (grow or stay same)"
            )

    @pytest.mark.vcr()
    def test_tool_call_events_have_consistent_tool_id(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that all events for the same tool call have the same tool ID."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="gpt-4o-mini", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        crew.kickoff()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 1,
                timeout=30,
            )

        assert success, "Timeout waiting for tool call streaming events"

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

    @pytest.mark.vcr()
    def test_tool_call_events_include_agent_and_task_info(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that tool call streaming events include agent and task context."""
        tool_call_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="gpt-4o-mini", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        crew.kickoff()

        with condition:
            success = condition.wait_for(
                lambda: len(tool_call_events) >= 1,
                timeout=30,
            )

        assert success, "Timeout waiting for tool call streaming events"

        for event in tool_call_events:
            assert event.agent_role == weather_agent.role
            assert event.agent_id == str(weather_agent.id)
            assert event.task_name is not None
            assert event.task_id is not None


class TestMixedStreamingEvents:
    """Tests for scenarios with both text and tool call streaming events."""

    @pytest.mark.vcr()
    def test_streaming_distinguishes_text_and_tool_calls(
        self, weather_agent: Agent, weather_task: Task
    ) -> None:
        """Test that streaming correctly distinguishes between text chunks and tool calls."""
        text_events: list[LLMStreamChunkEvent] = []
        tool_call_events: list[LLMStreamChunkEvent] = []
        all_events: list[LLMStreamChunkEvent] = []
        condition = threading.Condition()

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def handle_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with condition:
                all_events.append(event)
                if event.call_type == LLMCallType.TOOL_CALL:
                    tool_call_events.append(event)
                elif event.chunk and not event.tool_call:
                    text_events.append(event)
                condition.notify()

        weather_agent.llm = LLM(model="gpt-4o-mini", stream=True)
        crew = Crew(agents=[weather_agent], tasks=[weather_task], verbose=False)
        crew.kickoff()

        with condition:
            success = condition.wait_for(
                lambda: len(all_events) >= 5,
                timeout=30,
            )

        assert success, "Timeout waiting for streaming events"

        for event in tool_call_events:
            assert event.call_type == LLMCallType.TOOL_CALL
            assert event.tool_call is not None

        for event in text_events:
            assert event.call_type != LLMCallType.TOOL_CALL or event.call_type is None
            assert event.chunk is not None