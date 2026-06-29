"""Tests for the public stream frame contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import ConversationMessageAddedEvent
from crewai.events.types.llm_events import LLMStreamChunkEvent, LLMThinkingChunkEvent
from crewai.events.types.tool_usage_events import ToolUsageStartedEvent
from crewai.flow.flow import Flow, start
from crewai.llms.base_llm import BaseLLM
from crewai.types.streaming import FlowStreamingOutput, StreamFrame


class FrameFlow(Flow):
    @start()
    def run(self) -> str:
        crewai_event_bus.emit(
            self,
            LLMStreamChunkEvent(
                type="llm_stream_chunk",
                chunk="hello",
                call_id="call-1",
            ),
        )
        crewai_event_bus.emit(
            self,
            LLMThinkingChunkEvent(
                type="llm_thinking_chunk",
                chunk="thinking",
                call_id="call-1",
            ),
        )
        crewai_event_bus.emit(
            self,
            ConversationMessageAddedEvent(
                type="conversation_message_added",
                flow_name=self._definition.name,
                session_id="session-1",
                role="assistant",
                content="hello",
                message_index=0,
            ),
        )
        crewai_event_bus.emit(
            self,
            ToolUsageStartedEvent(
                type="tool_usage_started",
                tool_name="search",
                tool_args={"query": "crew"},
            ),
        )
        return "done"


class DirectStreamingLLM(BaseLLM):
    def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
        crewai_event_bus.emit(
            self,
            LLMStreamChunkEvent(
                type="llm_stream_chunk",
                chunk="hel",
                call_id="call-1",
            ),
        )
        crewai_event_bus.emit(
            self,
            LLMStreamChunkEvent(
                type="llm_stream_chunk",
                chunk="lo",
                call_id="call-1",
            ),
        )
        return "hello"


def test_stream_frame_contract_and_ordering() -> None:
    stream = FrameFlow().stream_events()

    with pytest.raises(RuntimeError, match="Streaming has not completed yet"):
        _ = stream.result

    with stream:
        frames = list(stream.events)

    assert stream.result == "done"
    assert all(isinstance(frame, StreamFrame) for frame in frames)
    assert all(frame.version == "v1" for frame in frames)
    assert [frame.seq for frame in frames] == sorted(frame.seq for frame in frames)

    by_type = {frame.type: frame for frame in frames}
    assert by_type["flow_started"].channel == "flow"
    assert by_type["method_execution_started"].parent_id == by_type["flow_started"].id
    assert by_type["llm_stream_chunk"].channel == "llm"
    assert by_type["llm_thinking_chunk"].channel == "llm"
    assert by_type["conversation_message_added"].channel == "messages"
    assert by_type["tool_usage_started"].channel == "tools"
    assert "FrameFlow" in by_type["method_execution_started"].namespace
    assert "run" in by_type["method_execution_started"].namespace


def test_stream_subscribe_filters_channels_without_losing_order() -> None:
    with FrameFlow().stream_events() as stream:
        frames = list(stream.interleave(["messages", "tools"]))

    assert [frame.channel for frame in frames] == ["messages", "tools"]
    assert [frame.seq for frame in frames] == sorted(frame.seq for frame in frames)
    assert stream.result == "done"


def test_stream_errors_surface_after_failed_frame() -> None:
    class ErrorFlow(Flow):
        @start()
        def run(self) -> str:
            raise ValueError("boom")

    stream = ErrorFlow().stream_events()

    with pytest.raises(ValueError, match="boom"):
        list(stream.events)

    assert any(frame.type == "method_execution_failed" for frame in stream.frames)
    with pytest.raises(ValueError, match="boom"):
        _ = stream.result


def test_legacy_flow_streaming_uses_llm_frame_projection() -> None:
    flow = FrameFlow()
    flow.stream = True

    streaming = flow.kickoff()

    assert isinstance(streaming, FlowStreamingOutput)
    chunks = list(streaming)
    assert [chunk.content for chunk in chunks] == ["hello", "thinking"]
    assert streaming.result == "done"


def test_direct_llm_stream_events_scope_and_restore_stream_flag() -> None:
    llm = DirectStreamingLLM(model="gpt-4o-mini", stream=False)

    with llm.stream_events("hello") as stream:
        frames = list(stream.llm)

    assert [frame.data["chunk"] for frame in frames] == ["hel", "lo"]
    assert stream.result == "hello"
    assert llm.stream is False


def test_direct_llm_stream_call_projects_chunks() -> None:
    chunks = DirectStreamingLLM(model="gpt-4o-mini").stream_call("hello")

    assert [chunk.content for chunk in chunks] == ["hel", "lo"]
    assert chunks.result == "hello"


@pytest.mark.asyncio
async def test_astream_scopes_concurrent_executions() -> None:
    class ConcurrentFlow(Flow):
        @start()
        async def run(self) -> str:
            label = str(self.state["label"])
            await asyncio.sleep(0)
            crewai_event_bus.emit(
                self,
                LLMStreamChunkEvent(
                    type="llm_stream_chunk",
                    chunk=label,
                    call_id=label,
                ),
            )
            return label

    async def collect(label: str) -> tuple[str, list[str]]:
        async with ConcurrentFlow().astream(inputs={"label": label}) as stream:
            frames = [frame async for frame in stream.llm]
        return stream.result, [frame.data["chunk"] for frame in frames]

    first, second = await asyncio.gather(collect("first"), collect("second"))

    assert first == ("first", ["first"])
    assert second == ("second", ["second"])


@pytest.mark.asyncio
async def test_astream_cancellation_cleans_up_task() -> None:
    class SlowFlow(Flow):
        @start()
        async def run(self) -> str:
            await asyncio.sleep(10)
            return "too late"

    stream = SlowFlow().astream()
    events: AsyncIterator[StreamFrame] = stream.events
    first_frame = await anext(events)

    assert first_frame.type == "flow_started"
    await stream.aclose()

    assert stream.is_cancelled is True
    assert stream.is_completed is True
