"""Streaming (SSE) update mechanism handler."""

from __future__ import annotations

import uuid

from a2a.client import Client
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

from crewai.a2a.task_helpers import TaskStateResult, process_task_state
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import A2AResponseReceivedEvent


async def execute_streaming_delegation(
    client: Client,
    message: Message,
    context_id: str | None,
    task_id: str | None,
    turn_number: int,
    is_multiturn: bool,
    agent_role: str | None,
    new_messages: list[Message],
    agent_card: AgentCard,
) -> TaskStateResult:
    """Execute A2A delegation using SSE streaming for updates.

    Args:
        client: A2A client instance
        message: Message to send
        context_id: Context ID for correlation
        task_id: Task ID for correlation
        turn_number: Current turn number
        is_multiturn: Whether this is a multi-turn conversation
        agent_role: Agent role for logging
        new_messages: List to collect messages
        agent_card: The agent card

    Returns:
        Dictionary with status, result/error, and history
    """
    result_parts: list[str] = []
    final_result: TaskStateResult | None = None
    event_stream = client.send_message(message)

    try:
        async for event in event_stream:
            if isinstance(event, Message):
                new_messages.append(event)
                for part in event.parts:
                    if part.root.kind == "text":
                        text = part.root.text
                        result_parts.append(text)

            elif isinstance(event, tuple):
                a2a_task, update = event

                if isinstance(update, TaskArtifactUpdateEvent):
                    artifact = update.artifact
                    result_parts.extend(
                        part.root.text
                        for part in artifact.parts
                        if part.root.kind == "text"
                    )

                is_final_update = False
                if isinstance(update, TaskStatusUpdateEvent):
                    is_final_update = update.final

                if not is_final_update and a2a_task.status.state not in [
                    TaskState.completed,
                    TaskState.input_required,
                    TaskState.failed,
                    TaskState.rejected,
                    TaskState.auth_required,
                    TaskState.canceled,
                ]:
                    continue

                final_result = process_task_state(
                    a2a_task=a2a_task,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    turn_number=turn_number,
                    is_multiturn=is_multiturn,
                    agent_role=agent_role,
                    result_parts=result_parts,
                )
                if final_result:
                    break

    except A2AClientHTTPError as e:
        error_msg = f"HTTP Error {e.status_code}: {e!s}"

        error_message = Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=error_msg))],
            context_id=context_id,
            task_id=task_id,
        )
        new_messages.append(error_message)

        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=error_msg,
                turn_number=turn_number,
                is_multiturn=is_multiturn,
                status="failed",
                agent_role=agent_role,
            ),
        )
        return TaskStateResult(
            status=TaskState.failed,
            error=error_msg,
            history=new_messages,
        )

    except Exception as e:
        current_exception: Exception | BaseException | None = e
        while current_exception:
            if hasattr(current_exception, "response"):
                response = current_exception.response
                if hasattr(response, "text"):
                    break
            if current_exception and hasattr(current_exception, "__cause__"):
                current_exception = current_exception.__cause__
        raise

    finally:
        if hasattr(event_stream, "aclose"):
            await event_stream.aclose()

    if final_result:
        return final_result

    return TaskStateResult(
        status=TaskState.completed,
        result=" ".join(result_parts) if result_parts else "",
        history=new_messages,
    )
