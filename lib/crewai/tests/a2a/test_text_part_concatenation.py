from __future__ import annotations

import uuid

import pytest

a2a_types = pytest.importorskip("a2a.types")
AgentCapabilities = a2a_types.AgentCapabilities
AgentCard = a2a_types.AgentCard
Artifact = a2a_types.Artifact
Message = a2a_types.Message
Part = a2a_types.Part
Role = a2a_types.Role
Task = a2a_types.Task
TaskArtifactUpdateEvent = a2a_types.TaskArtifactUpdateEvent
TaskState = a2a_types.TaskState
TaskStatus = a2a_types.TaskStatus
TaskStatusUpdateEvent = a2a_types.TaskStatusUpdateEvent
TextPart = a2a_types.TextPart

from crewai.a2a.task_helpers import process_task_state, send_message_and_get_task_id
from crewai.a2a.updates.streaming.handler import StreamingHandler
from crewai.a2a.wrapper import _handle_max_turns_exceeded


def _text_part(text: str) -> Part:
    return Part(root=TextPart(text=text))


def _agent_card() -> AgentCard:
    return AgentCard(
        name="Chunking agent",
        description="A test A2A agent",
        url="http://localhost:9999",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


async def _streaming_events(chunks: list[str]):
    task_id = "task-1"
    context_id = "context-1"
    task = Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=TaskState.working),
    )
    for index, chunk in enumerate(chunks):
        yield (
            task,
            TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=Artifact(artifact_id="reply", parts=[_text_part(chunk)]),
                append=index > 0,
                last_chunk=index == len(chunks) - 1,
            ),
        )
    yield (
        task,
        TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.completed),
            final=True,
        ),
    )


class _StreamingClient:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def send_message(self, _message: Message):
        return _streaming_events(self.chunks)


@pytest.mark.asyncio
async def test_streaming_artifact_text_chunks_concatenate_without_spaces() -> None:
    chunks = ["Hel", "lo, ", "wor", "ld", "!"]

    result = await StreamingHandler.execute(
        client=_StreamingClient(chunks),
        message=Message(
            role=Role.user,
            parts=[_text_part("say hello")],
            message_id=str(uuid.uuid4()),
        ),
        new_messages=[],
        agent_card=_agent_card(),
        endpoint="http://localhost:9999",
    )

    assert result["result"] == "Hello, world!"


def test_completed_task_artifact_text_chunks_concatenate_without_spaces() -> None:
    task = Task(
        id="task-1",
        context_id="context-1",
        status=TaskStatus(state=TaskState.completed),
        artifacts=[
            Artifact(artifact_id="reply", parts=[_text_part("Hel")]),
            Artifact(artifact_id="reply", parts=[_text_part("lo, ")]),
            Artifact(artifact_id="reply", parts=[_text_part("world")]),
        ],
    )

    result = process_task_state(
        task,
        new_messages=[],
        agent_card=_agent_card(),
        turn_number=1,
        is_multiturn=False,
        agent_role=None,
    )

    assert result is not None
    assert result["result"] == "Hello, world"


@pytest.mark.asyncio
async def test_immediate_message_text_parts_concatenate_without_spaces() -> None:
    async def events():
        yield Message(
            role=Role.agent,
            parts=[_text_part("Hel"), _text_part("lo")],
            message_id=str(uuid.uuid4()),
            context_id="context-1",
        )

    result = await send_message_and_get_task_id(
        events(),
        new_messages=[],
        agent_card=_agent_card(),
        turn_number=1,
        is_multiturn=False,
        agent_role=None,
    )

    assert isinstance(result, dict)
    assert result["result"] == "Hello"


def test_max_turns_fallback_text_parts_concatenate_without_spaces() -> None:
    result = _handle_max_turns_exceeded(
        [
            Message(
                role=Role.agent,
                parts=[_text_part("Hel"), _text_part("lo")],
                message_id=str(uuid.uuid4()),
            )
        ],
        max_turns=1,
    )

    assert result == "Hello"
