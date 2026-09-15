"""Tests for how the streaming handler joins artifact text.

A server that streams its reply sends each chunk as an artifact update with
``append=True``. Those chunks are one piece of text, so they are joined with no
separator, while separate artifacts keep their separator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
import uuid

from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
import pytest

from crewai.a2a.updates.streaming.handler import StreamingHandler


TASK_ID = "task-1"
CONTEXT_ID = "ctx-1"

Event = tuple[Task, TaskArtifactUpdateEvent | TaskStatusUpdateEvent]


def _task(state: TaskState) -> Task:
    return Task(id=TASK_ID, context_id=CONTEXT_ID, status=TaskStatus(state=state))


def _chunk(artifact_id: str, text: str, append: bool) -> Event:
    return (
        _task(TaskState.working),
        TaskArtifactUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            artifact=Artifact(
                artifact_id=artifact_id, parts=[Part(root=TextPart(text=text))]
            ),
            append=append,
        ),
    )


def _completed() -> Event:
    return (
        _task(TaskState.completed),
        TaskStatusUpdateEvent(
            task_id=TASK_ID,
            context_id=CONTEXT_ID,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ),
    )


async def _stream(
    events: Sequence[Event], error: Exception | None = None
) -> AsyncIterator[Event]:
    for event in events:
        yield event
    if error is not None:
        raise error


class _FakeClient:
    """Replays a scripted stream in place of ``a2a.client.Client``."""

    def __init__(
        self,
        events: Sequence[Event],
        error: Exception | None = None,
        resubscribe_events: Sequence[Event] = (),
    ) -> None:
        self._events = events
        self._error = error
        self._resubscribe_events = resubscribe_events

    def send_message(self, message: Message) -> AsyncIterator[Event]:
        return _stream(self._events, self._error)

    async def get_task(self, params: object) -> Task:
        return _task(TaskState.working)

    def resubscribe(self, params: object) -> AsyncIterator[Event]:
        return _stream(self._resubscribe_events)


async def _run(client: _FakeClient) -> str:
    agent_card = AgentCard(
        name="Streaming Agent",
        description="Streams its reply in chunks.",
        url="http://localhost:9999",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )
    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text="say hello"))],
        message_id=str(uuid.uuid4()),
    )
    result = await StreamingHandler.execute(
        client=client,  # type: ignore[arg-type]
        message=message,
        new_messages=[],
        agent_card=agent_card,
        endpoint=agent_card.url,
    )
    return str(result["result"])


@pytest.mark.asyncio
async def test_appended_chunks_are_joined_without_spaces() -> None:
    """Chunks of one streamed artifact come back as the text the server sent."""
    chunks = ["Hel", "lo, ", "wor", "ld", "!"]
    events = [
        _chunk("reply", text, append=index > 0) for index, text in enumerate(chunks)
    ]

    assert await _run(_FakeClient(events)) == "Hello, world!"


@pytest.mark.asyncio
async def test_separate_artifacts_stay_separated() -> None:
    """Different artifacts are still joined with a space, as before."""
    events = [
        _chunk("summary", "First paragraph.", append=False),
        _chunk("details", "Second paragraph.", append=False),
        _completed(),
    ]

    assert await _run(_FakeClient(events)) == "First paragraph. Second paragraph."


@pytest.mark.asyncio
async def test_interleaved_artifacts_append_to_their_own_text() -> None:
    """An appended chunk joins its own artifact, not whichever arrived last."""
    events = [
        _chunk("greeting", "Hel", append=False),
        _chunk("subject", "Wor", append=False),
        _chunk("greeting", "lo", append=True),
        _chunk("subject", "ld", append=True),
        _completed(),
    ]

    assert await _run(_FakeClient(events)) == "Hello World"


@pytest.mark.asyncio
async def test_chunks_after_a_reconnect_continue_the_same_text() -> None:
    """After the stream drops and resubscribes, appended chunks keep joining."""
    client = _FakeClient(
        events=[_chunk("reply", "Hello, ", append=False)],
        error=A2AClientHTTPError(503, "stream dropped"),
        resubscribe_events=[
            _chunk("reply", "world", append=True),
            _chunk("reply", "!", append=True),
            _completed(),
        ],
    )

    assert await _run(client) == "Hello, world!"
