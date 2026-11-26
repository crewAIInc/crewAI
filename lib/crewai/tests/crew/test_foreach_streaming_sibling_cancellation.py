"""Regression tests: run_for_each_async streaming path must cancel sibling
stream consumers when one of them fails, instead of abandoning them to run in
the background after the caller already holds the error.
"""

import asyncio

import pytest

from crewai import Agent, Crew, Task
from crewai.crews.utils import run_for_each_async
from crewai.types.streaming import CrewStreamingOutput, StreamChunk


def _make_crew() -> Crew:
    agent = Agent(role="R", goal="G", backstory="B", allow_delegation=False)
    task = Task(description="d", expected_output="o", agent=agent)
    return Crew(agents=[agent], tasks=[task], stream=True)


async def _consume(streaming: CrewStreamingOutput) -> tuple[list[str], Exception | None]:
    chunks: list[str] = []
    error: Exception | None = None
    try:
        async for chunk in streaming:
            chunks.append(chunk.content)
    except Exception as exc:  # noqa: BLE001
        error = exc
    return chunks, error


@pytest.mark.asyncio
async def test_failing_consumer_cancels_sibling_consumer() -> None:
    """A mid-stream failure must cancel the in-flight sibling before the error
    reaches the caller (fails on main: sibling is left running in background)."""
    cancelled = asyncio.Event()
    slow_finished: list[str] = []

    async def slow_stream():
        try:
            yield StreamChunk(content="a1")
            await asyncio.sleep(2)
            slow_finished.append("slow finished after the error surfaced")
            yield StreamChunk(content="a2")
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def failing_stream():
        yield StreamChunk(content="b1")
        await asyncio.sleep(0.05)
        raise RuntimeError("stream B exploded mid-stream")

    outputs = [
        CrewStreamingOutput(async_iterator=slow_stream()),
        CrewStreamingOutput(async_iterator=failing_stream()),
    ]

    async def fake_kickoff(_crew_copy: Crew, _inputs: dict) -> CrewStreamingOutput:
        return outputs.pop(0)

    streaming = await run_for_each_async(_make_crew(), [{}, {}], fake_kickoff)

    chunks, error = await _consume(streaming)

    assert error is not None
    assert isinstance(error, RuntimeError)
    assert "b1" in chunks
    assert cancelled.is_set(), "sibling consumer should be cancelled on failure"
    assert not slow_finished, "sibling should not run to completion after the error"


@pytest.mark.asyncio
async def test_successful_streams_all_run_to_completion() -> None:
    """Guard: without failures all stream consumers complete and their chunks flow."""
    done: list[str] = []

    def make_stream(tag: str):
        async def stream():
            yield StreamChunk(content=f"{tag}1")
            await asyncio.sleep(0.05)
            yield StreamChunk(content=f"{tag}2")
            done.append(tag)

        return stream()

    outputs = [
        CrewStreamingOutput(async_iterator=make_stream("a")),
        CrewStreamingOutput(async_iterator=make_stream("b")),
    ]
    for output in outputs:
        # The real chunk generator sets results; fake streams set them directly.
        output._set_result("result")  # type: ignore[arg-type]

    async def fake_kickoff(_crew_copy: Crew, _inputs: dict) -> CrewStreamingOutput:
        return outputs.pop(0)

    streaming = await run_for_each_async(_make_crew(), [{}, {}], fake_kickoff)

    chunks, error = await _consume(streaming)

    assert error is None
    assert sorted(chunks) == ["a1", "a2", "b1", "b2"]
    assert sorted(done) == ["a", "b"]
