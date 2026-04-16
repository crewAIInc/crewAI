"""Tests for streaming output functionality in crews and flows."""

import asyncio
from collections.abc import AsyncIterator, Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMStreamChunkEvent, ToolCall, FunctionCall
from crewai.flow.flow import Flow, start
from crewai.types.streaming import (
    CrewStreamingOutput,
    FlowStreamingOutput,
    StreamChunk,
    StreamChunkType,
    ToolCallChunk,
)


@pytest.fixture
def researcher() -> Agent:
    """Create a researcher agent for testing."""
    return Agent(
        role="Researcher",
        goal="Research and analyze topics thoroughly",
        backstory="You are an expert researcher with deep analytical skills.",
        allow_delegation=False,
    )


@pytest.fixture
def simple_task(researcher: Agent) -> Task:
    """Create a simple task for testing."""
    return Task(
        description="Write a brief analysis of AI trends",
        expected_output="A concise analysis of current AI trends",
        agent=researcher,
    )


@pytest.fixture
def simple_crew(researcher: Agent, simple_task: Task) -> Crew:
    """Create a simple crew with one agent and one task."""
    return Crew(
        agents=[researcher],
        tasks=[simple_task],
        verbose=False,
    )


@pytest.fixture
def streaming_crew(researcher: Agent, simple_task: Task) -> Crew:
    """Create a streaming crew with one agent and one task."""
    return Crew(
        agents=[researcher],
        tasks=[simple_task],
        verbose=False,
        stream=True,
    )


class TestStreamChunk:
    """Tests for StreamChunk model."""

    def test_stream_chunk_creation(self) -> None:
        """Test creating a basic stream chunk."""
        chunk = StreamChunk(
            content="Hello, world!",
            chunk_type=StreamChunkType.TEXT,
            task_index=0,
            task_name="Test Task",
            task_id="task-123",
            agent_role="Researcher",
            agent_id="agent-456",
        )
        assert chunk.content == "Hello, world!"
        assert chunk.chunk_type == StreamChunkType.TEXT
        assert chunk.task_index == 0
        assert chunk.task_name == "Test Task"
        assert str(chunk) == "Hello, world!"

    def test_stream_chunk_with_tool_call(self) -> None:
        """Test creating a stream chunk with tool call information."""
        tool_call = ToolCallChunk(
            tool_id="call-123",
            tool_name="search",
            arguments='{"query": "AI trends"}',
            index=0,
        )
        chunk = StreamChunk(
            content="",
            chunk_type=StreamChunkType.TOOL_CALL,
            tool_call=tool_call,
        )
        assert chunk.chunk_type == StreamChunkType.TOOL_CALL
        assert chunk.tool_call is not None
        assert chunk.tool_call.tool_name == "search"


class TestCrewStreamingOutput:
    """Tests for CrewStreamingOutput functionality."""

    def test_result_before_iteration_raises_error(self) -> None:
        """Test that accessing result before iteration raises error."""

        def empty_gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="test")

        streaming = CrewStreamingOutput(sync_iterator=empty_gen())
        with pytest.raises(RuntimeError, match="Streaming has not completed yet"):
            _ = streaming.result

    def test_is_completed_property(self) -> None:
        """Test the is_completed property."""

        def simple_gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="test")

        streaming = CrewStreamingOutput(sync_iterator=simple_gen())
        assert streaming.is_completed is False

        list(streaming)
        assert streaming.is_completed is True

    def test_get_full_text(self) -> None:
        """Test getting full text from chunks."""

        def gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="Hello ")
            yield StreamChunk(content="World!")
            yield StreamChunk(content="", chunk_type=StreamChunkType.TOOL_CALL)

        streaming = CrewStreamingOutput(sync_iterator=gen())
        list(streaming)
        assert streaming.get_full_text() == "Hello World!"

    def test_chunks_property(self) -> None:
        """Test accessing collected chunks."""

        def gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="chunk1")
            yield StreamChunk(content="chunk2")

        streaming = CrewStreamingOutput(sync_iterator=gen())
        list(streaming)
        assert len(streaming.chunks) == 2
        assert streaming.chunks[0].content == "chunk1"


class TestFlowStreamingOutput:
    """Tests for FlowStreamingOutput functionality."""

    def test_result_before_iteration_raises_error(self) -> None:
        """Test that accessing result before iteration raises error."""

        def empty_gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="test")

        streaming = FlowStreamingOutput(sync_iterator=empty_gen())
        with pytest.raises(RuntimeError, match="Streaming has not completed yet"):
            _ = streaming.result

    def test_is_completed_property(self) -> None:
        """Test the is_completed property."""

        def simple_gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="test")

        streaming = FlowStreamingOutput(sync_iterator=simple_gen())
        assert streaming.is_completed is False

        list(streaming)
        assert streaming.is_completed is True


class TestCrewKickoffStreaming:
    """Tests for Crew(stream=True).kickoff() method."""

    def test_kickoff_streaming_returns_streaming_output(self, streaming_crew: Crew) -> None:
        """Test that kickoff with stream=True returns CrewStreamingOutput."""
        with patch.object(Crew, "kickoff") as mock_kickoff:
            mock_output = MagicMock()
            mock_output.raw = "Test output"

            def side_effect(*args: Any, **kwargs: Any) -> Any:
                return mock_output
            mock_kickoff.side_effect = side_effect

        streaming = streaming_crew.kickoff()
        assert isinstance(streaming, CrewStreamingOutput)

    def test_kickoff_streaming_captures_chunks(self, researcher: Agent, simple_task: Task) -> None:
        """Test that streaming captures LLM chunks."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            verbose=False,
            stream=True,
        )

        mock_output = MagicMock()
        mock_output.raw = "Test output"

        original_kickoff = Crew.kickoff
        call_count = [0]

        def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_kickoff(self, inputs, **kwargs)
            else:
                crewai_event_bus.emit(
                    crew,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="Hello ",
                        call_id="test-call-id",
                    ),
                )
                crewai_event_bus.emit(
                    crew,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="World!",
                        call_id="test-call-id",
                    ),
                )
                return mock_output

        with patch.object(Crew, "kickoff", mock_kickoff_fn):
            streaming = crew.kickoff()
            assert isinstance(streaming, CrewStreamingOutput)
            chunks = list(streaming)

        assert len(chunks) >= 2
        contents = [c.content for c in chunks]
        assert "Hello " in contents
        assert "World!" in contents

    def test_kickoff_streaming_result_available_after_iteration(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test that result is available after iterating all chunks."""
        mock_output = MagicMock()
        mock_output.raw = "Final result"

        def gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="test chunk")

        streaming = CrewStreamingOutput(sync_iterator=gen())

        # Iterate all chunks
        _ = list(streaming)

        # Simulate what _finalize_streaming does
        streaming._set_result(mock_output)

        result = streaming.result
        assert result.raw == "Final result"

    def test_kickoff_streaming_handles_tool_calls(self, researcher: Agent, simple_task: Task) -> None:
        """Test that streaming handles tool call chunks correctly."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            verbose=False,
            stream=True,
        )

        mock_output = MagicMock()
        mock_output.raw = "Test output"

        original_kickoff = Crew.kickoff
        call_count = [0]

        def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_kickoff(self, inputs, **kwargs)
            else:
                crewai_event_bus.emit(
                    crew,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="",
                        call_id="test-call-id",
                        tool_call=ToolCall(
                            id="call-123",
                            function=FunctionCall(
                                name="search",
                                arguments='{"query": "test"}',
                            ),
                            type="function",
                            index=0,
                        ),
                    ),
                )
                return mock_output

        with patch.object(Crew, "kickoff", mock_kickoff_fn):
            streaming = crew.kickoff()
            assert isinstance(streaming, CrewStreamingOutput)
            chunks = list(streaming)

        tool_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.TOOL_CALL]
        assert len(tool_chunks) >= 1
        assert tool_chunks[0].tool_call is not None
        assert tool_chunks[0].tool_call.tool_name == "search"


class TestCrewKickoffStreamingAsync:
    """Tests for Crew(stream=True).kickoff_async() method."""

    @pytest.mark.asyncio
    async def test_kickoff_streaming_async_returns_streaming_output(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test that kickoff_async with stream=True returns CrewStreamingOutput."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            verbose=False,
            stream=True,
        )

        mock_output = MagicMock()
        mock_output.raw = "Test output"

        original_kickoff = Crew.kickoff
        call_count = [0]

        def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_kickoff(self, inputs, **kwargs)
            else:
                return mock_output

        with patch.object(Crew, "kickoff", mock_kickoff_fn):
            streaming = await crew.kickoff_async()

        assert isinstance(streaming, CrewStreamingOutput)

    @pytest.mark.asyncio
    async def test_kickoff_streaming_async_captures_chunks(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test that async streaming captures LLM chunks."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            verbose=False,
            stream=True,
        )

        mock_output = MagicMock()
        mock_output.raw = "Test output"

        def mock_kickoff_fn(
            self: Any, inputs: Any = None, input_files: Any = None, **kwargs: Any
        ) -> Any:
            crewai_event_bus.emit(
                crew,
                LLMStreamChunkEvent(
                    type="llm_stream_chunk",
                    chunk="Async ",
                    call_id="test-call-id",
                ),
            )
            crewai_event_bus.emit(
                crew,
                LLMStreamChunkEvent(
                    type="llm_stream_chunk",
                    chunk="Stream!",
                    call_id="test-call-id",
                ),
            )
            return mock_output

        with patch.object(Crew, "kickoff", mock_kickoff_fn):
            streaming = await crew.kickoff_async()
            assert isinstance(streaming, CrewStreamingOutput)
            chunks: list[StreamChunk] = []
            async for chunk in streaming:
                chunks.append(chunk)

        assert len(chunks) >= 2
        contents = [c.content for c in chunks]
        assert "Async " in contents
        assert "Stream!" in contents

    @pytest.mark.asyncio
    async def test_kickoff_streaming_async_result_available_after_iteration(
        self, researcher: Agent, simple_task: Task
    ) -> None:
        """Test that result is available after async iteration."""
        mock_output = MagicMock()
        mock_output.raw = "Async result"

        async def async_gen() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="test chunk")

        streaming = CrewStreamingOutput(async_iterator=async_gen())

        # Iterate all chunks
        async for _ in streaming:
            pass

        # Simulate what _finalize_streaming does
        streaming._set_result(mock_output)

        result = streaming.result
        assert result.raw == "Async result"


class TestFlowKickoffStreaming:
    """Tests for Flow(stream=True).kickoff() method."""

    def test_kickoff_streaming_returns_streaming_output(self) -> None:
        """Test that flow kickoff with stream=True returns FlowStreamingOutput."""

        class SimpleFlow(Flow[dict[str, Any]]):
            @start()
            def generate(self) -> str:
                return "result"

        flow = SimpleFlow()
        flow.stream = True
        streaming = flow.kickoff()
        assert isinstance(streaming, FlowStreamingOutput)

    def test_flow_kickoff_streaming_captures_chunks(self) -> None:
        """Test that flow streaming captures LLM chunks from crew execution."""

        class TestFlow(Flow[dict[str, Any]]):
            @start()
            def run_crew(self) -> str:
                return "done"

        flow = TestFlow()
        flow.stream = True

        original_kickoff = Flow.kickoff
        call_count = [0]

        def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_kickoff(self, inputs, **kwargs)
            else:
                crewai_event_bus.emit(
                    flow,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="Flow ",
                        call_id="test-call-id",
                    ),
                )
                crewai_event_bus.emit(
                    flow,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="output!",
                        call_id="test-call-id",
                    ),
                )
                return "done"

        with patch.object(Flow, "kickoff", mock_kickoff_fn):
            streaming = flow.kickoff()
            assert isinstance(streaming, FlowStreamingOutput)
            chunks = list(streaming)

        assert len(chunks) >= 2
        contents = [c.content for c in chunks]
        assert "Flow " in contents
        assert "output!" in contents

    def test_flow_kickoff_streaming_result_available(self) -> None:
        """Test that flow result is available after iteration."""

        class TestFlow(Flow[dict[str, Any]]):
            @start()
            def generate(self) -> str:
                return "flow result"

        flow = TestFlow()
        flow.stream = True

        original_kickoff = Flow.kickoff
        call_count = [0]

        def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_kickoff(self, inputs, **kwargs)
            else:
                return "flow result"

        with patch.object(Flow, "kickoff", mock_kickoff_fn):
            streaming = flow.kickoff()
            assert isinstance(streaming, FlowStreamingOutput)
            _ = list(streaming)

        result = streaming.result
        assert result == "flow result"


class TestFlowKickoffStreamingAsync:
    """Tests for Flow(stream=True).kickoff_async() method."""

    @pytest.mark.asyncio
    async def test_kickoff_streaming_async_returns_streaming_output(self) -> None:
        """Test that flow kickoff_async with stream=True returns FlowStreamingOutput."""

        class SimpleFlow(Flow[dict[str, Any]]):
            @start()
            async def generate(self) -> str:
                return "async result"

        flow = SimpleFlow()
        flow.stream = True
        streaming = await flow.kickoff_async()
        assert isinstance(streaming, FlowStreamingOutput)

    @pytest.mark.asyncio
    async def test_flow_kickoff_streaming_async_captures_chunks(self) -> None:
        """Test that async flow streaming captures LLM chunks."""

        class TestFlow(Flow[dict[str, Any]]):
            @start()
            async def run_crew(self) -> str:
                return "done"

        flow = TestFlow()
        flow.stream = True

        original_kickoff = Flow.kickoff_async
        call_count = [0]

        async def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return await original_kickoff(self, inputs, **kwargs)
            else:
                await asyncio.sleep(0.01)
                crewai_event_bus.emit(
                    flow,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="Async flow ",
                        call_id="test-call-id",
                    ),
                )
                await asyncio.sleep(0.01)
                crewai_event_bus.emit(
                    flow,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="stream!",
                        call_id="test-call-id",
                    ),
                )
                await asyncio.sleep(0.01)
                return "done"

        with patch.object(Flow, "kickoff_async", mock_kickoff_fn):
            streaming = await flow.kickoff_async()
            assert isinstance(streaming, FlowStreamingOutput)
            chunks: list[StreamChunk] = []
            async for chunk in streaming:
                chunks.append(chunk)

        assert len(chunks) >= 2
        contents = [c.content for c in chunks]
        assert "Async flow " in contents
        assert "stream!" in contents

    @pytest.mark.asyncio
    async def test_flow_kickoff_streaming_async_result_available(self) -> None:
        """Test that async flow result is available after iteration."""

        class TestFlow(Flow[dict[str, Any]]):
            @start()
            async def generate(self) -> str:
                return "async flow result"

        flow = TestFlow()
        flow.stream = True

        original_kickoff = Flow.kickoff_async
        call_count = [0]

        async def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return await original_kickoff(self, inputs, **kwargs)
            else:
                return "async flow result"

        with patch.object(Flow, "kickoff_async", mock_kickoff_fn):
            streaming = await flow.kickoff_async()
            assert isinstance(streaming, FlowStreamingOutput)
            async for _ in streaming:
                pass

        result = streaming.result
        assert result == "async flow result"


class TestStreamingEdgeCases:
    """Tests for edge cases in streaming functionality."""

    def test_streaming_handles_exceptions(self, researcher: Agent, simple_task: Task) -> None:
        """Test that streaming properly propagates exceptions."""
        crew = Crew(
            agents=[researcher],
            tasks=[simple_task],
            verbose=False,
            stream=True,
        )

        original_kickoff = Crew.kickoff
        call_count = [0]

        def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_kickoff(self, inputs, **kwargs)
            else:
                raise ValueError("Test error")

        with patch.object(Crew, "kickoff", mock_kickoff_fn):
            streaming = crew.kickoff()
            with pytest.raises(ValueError, match="Test error"):
                list(streaming)

    def test_streaming_with_empty_content_chunks(self) -> None:
        """Test streaming when LLM chunks have empty content."""
        mock_output = MagicMock()
        mock_output.raw = "No streaming"

        def gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="")

        streaming = CrewStreamingOutput(sync_iterator=gen())
        chunks = list(streaming)

        assert streaming.is_completed
        assert len(chunks) == 1
        assert chunks[0].content == ""

        # Simulate what _finalize_streaming does
        streaming._set_result(mock_output)

        result = streaming.result
        assert result.raw == "No streaming"

    def test_streaming_with_multiple_tasks(self, researcher: Agent) -> None:
        """Test streaming with multiple tasks tracks task context."""
        task1 = Task(
            description="First task",
            expected_output="First output",
            agent=researcher,
        )
        task2 = Task(
            description="Second task",
            expected_output="Second output",
            agent=researcher,
        )
        crew = Crew(
            agents=[researcher],
            tasks=[task1, task2],
            verbose=False,
            stream=True,
        )

        mock_output = MagicMock()
        mock_output.raw = "Multi-task output"

        original_kickoff = Crew.kickoff
        call_count = [0]

        def mock_kickoff_fn(self: Any, inputs: Any = None, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return original_kickoff(self, inputs, **kwargs)
            else:
                crewai_event_bus.emit(
                    crew,
                    LLMStreamChunkEvent(
                        type="llm_stream_chunk",
                        chunk="Task 1",
                        task_name="First task",
                        call_id="test-call-id",
                    ),
                )
                return mock_output

        with patch.object(Crew, "kickoff", mock_kickoff_fn):
            streaming = crew.kickoff()
            assert isinstance(streaming, CrewStreamingOutput)
            chunks = list(streaming)

        assert len(chunks) >= 1
        assert streaming.is_completed


class TestStreamingCancellation:
    """Tests for streaming cancellation and resource cleanup."""

    @pytest.mark.asyncio
    async def test_aclose_cancels_async_streaming(self) -> None:
        """Test that aclose() stops iteration and marks as cancelled."""
        chunks_yielded: list[str] = []

        async def slow_gen() -> AsyncIterator[StreamChunk]:
            for i in range(100):
                await asyncio.sleep(0.01)
                chunks_yielded.append(f"chunk-{i}")
                yield StreamChunk(content=f"chunk-{i}")

        streaming = CrewStreamingOutput(async_iterator=slow_gen())
        collected: list[StreamChunk] = []

        async for chunk in streaming:
            collected.append(chunk)
            if len(collected) >= 3:
                break

        await streaming.aclose()

        assert streaming.is_cancelled
        assert streaming.is_completed
        assert len(collected) == 3

    @pytest.mark.asyncio
    async def test_aclose_idempotent(self) -> None:
        """Test that calling aclose() multiple times is safe."""
        async def gen() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="test")

        streaming = CrewStreamingOutput(async_iterator=gen())
        async for _ in streaming:
            pass

        await streaming.aclose()
        await streaming.aclose()
        assert not streaming.is_cancelled
        assert streaming.is_completed

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test using streaming output as async context manager."""
        async def gen() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="hello")
            yield StreamChunk(content="world")

        streaming = CrewStreamingOutput(async_iterator=gen())
        collected: list[StreamChunk] = []

        async with streaming:
            async for chunk in streaming:
                collected.append(chunk)

        assert not streaming.is_cancelled
        assert streaming.is_completed
        assert len(collected) == 2

    @pytest.mark.asyncio
    async def test_async_context_manager_early_exit(self) -> None:
        """Test context manager cleans up on early exit."""
        async def gen() -> AsyncIterator[StreamChunk]:
            for i in range(100):
                await asyncio.sleep(0.01)
                yield StreamChunk(content=f"chunk-{i}")

        streaming = CrewStreamingOutput(async_iterator=gen())

        async with streaming:
            async for chunk in streaming:
                if chunk.content == "chunk-2":
                    break

        assert streaming.is_cancelled
        assert streaming.is_completed

    def test_close_cancels_sync_streaming(self) -> None:
        """Test that close() stops sync streaming and marks as cancelled."""
        def gen() -> Generator[StreamChunk, None, None]:
            for i in range(100):
                yield StreamChunk(content=f"chunk-{i}")

        streaming = CrewStreamingOutput(sync_iterator=gen())
        collected: list[StreamChunk] = []

        for chunk in streaming:
            collected.append(chunk)
            if len(collected) >= 3:
                break

        streaming.close()

        assert streaming.is_cancelled
        assert streaming.is_completed

    def test_close_idempotent(self) -> None:
        """Test that calling close() multiple times is safe."""
        def gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="test")

        streaming = CrewStreamingOutput(sync_iterator=gen())
        list(streaming)

        streaming.close()
        streaming.close()
        assert not streaming.is_cancelled
        assert streaming.is_completed

    @pytest.mark.asyncio
    async def test_flow_aclose(self) -> None:
        """Test that FlowStreamingOutput aclose() is no-op after normal completion."""
        async def gen() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="flow-chunk")

        streaming = FlowStreamingOutput(async_iterator=gen())
        async for _ in streaming:
            pass

        await streaming.aclose()
        assert not streaming.is_cancelled
        assert streaming.is_completed

    @pytest.mark.asyncio
    async def test_flow_async_context_manager(self) -> None:
        """Test FlowStreamingOutput as async context manager with full consumption."""
        async def gen() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="flow-chunk")

        streaming = FlowStreamingOutput(async_iterator=gen())

        async with streaming:
            async for _ in streaming:
                pass

        assert not streaming.is_cancelled
        assert streaming.is_completed

    def test_flow_close(self) -> None:
        """Test that FlowStreamingOutput close() is no-op after normal completion."""
        def gen() -> Generator[StreamChunk, None, None]:
            yield StreamChunk(content="flow-chunk")

        streaming = FlowStreamingOutput(sync_iterator=gen())
        list(streaming)

        streaming.close()
        assert not streaming.is_cancelled


class TestStreamingImports:
    """Tests for correct imports of streaming types."""

    def test_streaming_types_importable_from_types_module(self) -> None:
        """Test that streaming types can be imported from crewai.types.streaming."""
        from crewai.types.streaming import (
            CrewStreamingOutput,
            FlowStreamingOutput,
            StreamChunk,
            StreamChunkType,
            ToolCallChunk,
        )

        assert CrewStreamingOutput is not None
        assert FlowStreamingOutput is not None
        assert StreamChunk is not None
        assert StreamChunkType is not None
        assert ToolCallChunk is not None


class TestConcurrentStreamIsolation:
    """Regression tests for concurrent streaming isolation (issue #5376)."""

    def test_concurrent_streams_do_not_cross_contaminate(self) -> None:
        """Two concurrent streaming runs must each receive only their own chunks.

        Mirrors the real production path: create_streaming_state in the caller,
        then temporarily push the stream_id into the ContextVar, copy_context,
        and reset — exactly as create_chunk_generator does.
        """
        import contextvars
        import threading

        from crewai.utilities.streaming import (
            TaskInfo,
            _current_stream_ids,
            _unregister_handler,
            create_streaming_state,
        )

        task_info_a: TaskInfo = {
            "index": 0,
            "name": "task_a",
            "id": "a",
            "agent_role": "A",
            "agent_id": "a",
        }
        task_info_b: TaskInfo = {
            "index": 1,
            "name": "task_b",
            "id": "b",
            "agent_role": "B",
            "agent_id": "b",
        }

        state_a = create_streaming_state(task_info_a, [])
        state_b = create_streaming_state(task_info_b, [])

        def make_emitter_ctx(state: Any) -> contextvars.Context:
            token = _current_stream_ids.set(
                (*_current_stream_ids.get(), state.stream_id)
            )
            ctx = contextvars.copy_context()
            _current_stream_ids.reset(token)
            return ctx

        ctx_a = make_emitter_ctx(state_a)
        ctx_b = make_emitter_ctx(state_b)

        def emit_chunks(prefix: str, call_id: str) -> None:
            for text in [f"{prefix}1", f"{prefix}2", f"{prefix}3"]:
                crewai_event_bus.emit(
                    None,
                    event=LLMStreamChunkEvent(
                        chunk=text, call_id=call_id, response_id="r"
                    ),
                )

        t_a = threading.Thread(target=ctx_a.run, args=(lambda: emit_chunks("A", "ca"),))
        t_b = threading.Thread(target=ctx_b.run, args=(lambda: emit_chunks("B", "cb"),))
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        chunks_a: list[str] = []
        while not state_a.sync_queue.empty():
            item = state_a.sync_queue.get_nowait()
            if isinstance(item, StreamChunk):
                chunks_a.append(item.content)

        chunks_b: list[str] = []
        while not state_b.sync_queue.empty():
            item = state_b.sync_queue.get_nowait()
            if isinstance(item, StreamChunk):
                chunks_b.append(item.content)

        assert set(chunks_a) == {"A1", "A2", "A3"}, (
            f"Stream A received unexpected chunks: {chunks_a}"
        )
        assert set(chunks_b) == {"B1", "B2", "B3"}, (
            f"Stream B received unexpected chunks: {chunks_b}"
        )

        _unregister_handler(state_a.handler)
        _unregister_handler(state_b.handler)
