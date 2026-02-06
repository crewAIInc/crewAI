"""Helper functions for processing A2A task results."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypedDict
import uuid

from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from typing_extensions import NotRequired

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConnectionErrorEvent,
    A2AResponseReceivedEvent,
)


if TYPE_CHECKING:
    from a2a.types import Task as A2ATask

SendMessageEvent = (
    tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None] | Message
)


TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.completed,
        TaskState.failed,
        TaskState.rejected,
        TaskState.canceled,
    }
)

ACTIONABLE_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.input_required,
        TaskState.auth_required,
    }
)

PENDING_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.submitted,
        TaskState.working,
    }
)


class TaskStateResult(TypedDict):
    """Result dictionary from processing A2A task state."""

    status: TaskState
    history: list[Message]
    result: NotRequired[str]
    error: NotRequired[str]
    agent_card: NotRequired[dict[str, Any]]
    a2a_agent_name: NotRequired[str | None]


def extract_task_result_parts(a2a_task: A2ATask) -> list[str]:
    """Extract result parts from A2A task status message, history, and artifacts.

    Args:
        a2a_task: A2A Task object with status, history, and artifacts

    Returns:
        List of result text parts
    """
    result_parts: list[str] = []

    if a2a_task.status and a2a_task.status.message:
        msg = a2a_task.status.message
        result_parts.extend(
            part.root.text for part in msg.parts if part.root.kind == "text"
        )

    if not result_parts and a2a_task.history:
        for history_msg in reversed(a2a_task.history):
            if history_msg.role == Role.agent:
                result_parts.extend(
                    part.root.text
                    for part in history_msg.parts
                    if part.root.kind == "text"
                )
                break

    if a2a_task.artifacts:
        result_parts.extend(
            part.root.text
            for artifact in a2a_task.artifacts
            for part in artifact.parts
            if part.root.kind == "text"
        )

    return result_parts


def extract_error_message(a2a_task: A2ATask, default: str) -> str:
    """Extract error message from A2A task.

    Args:
        a2a_task: A2A Task object
        default: Default message if no error found

    Returns:
        Error message string
    """
    if a2a_task.status and a2a_task.status.message:
        msg = a2a_task.status.message
        if msg:
            for part in msg.parts:
                if part.root.kind == "text":
                    return str(part.root.text)
        return str(msg)

    if a2a_task.history:
        for history_msg in reversed(a2a_task.history):
            for part in history_msg.parts:
                if part.root.kind == "text":
                    return str(part.root.text)

    return default


def process_task_state(
    a2a_task: A2ATask,
    new_messages: list[Message],
    agent_card: AgentCard,
    turn_number: int,
    is_multiturn: bool,
    agent_role: str | None,
    result_parts: list[str] | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    is_final: bool = True,
) -> TaskStateResult | None:
    """Process A2A task state and return result dictionary.

    Shared logic for both polling and streaming handlers.

    Args:
        a2a_task: The A2A task to process.
        new_messages: List to collect messages (modified in place).
        agent_card: The agent card.
        turn_number: Current turn number.
        is_multiturn: Whether multi-turn conversation.
        agent_role: Agent role for logging.
        result_parts: Accumulated result parts (streaming passes accumulated,
            polling passes None to extract from task).
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.
        from_task: Optional CrewAI Task for event metadata.
        from_agent: Optional CrewAI Agent for event metadata.
        is_final: Whether this is the final response in the stream.

    Returns:
        Result dictionary if terminal/actionable state, None otherwise.
    """
    if result_parts is None:
        result_parts = []

    if a2a_task.status.state == TaskState.completed:
        if not result_parts:
            extracted_parts = extract_task_result_parts(a2a_task)
            result_parts.extend(extracted_parts)
        if a2a_task.history:
            new_messages.extend(a2a_task.history)

        response_text = " ".join(result_parts) if result_parts else ""
        message_id = None
        if a2a_task.status and a2a_task.status.message:
            message_id = a2a_task.status.message.message_id
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=response_text,
                turn_number=turn_number,
                context_id=a2a_task.context_id,
                message_id=message_id,
                is_multiturn=is_multiturn,
                status="completed",
                final=is_final,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

        return TaskStateResult(
            status=TaskState.completed,
            agent_card=agent_card.model_dump(exclude_none=True),
            result=response_text,
            history=new_messages,
        )

    if a2a_task.status.state == TaskState.input_required:
        if a2a_task.history:
            new_messages.extend(a2a_task.history)

        response_text = extract_error_message(a2a_task, "Additional input required")
        if response_text and not a2a_task.history:
            agent_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=response_text))],
                context_id=a2a_task.context_id,
                task_id=a2a_task.id,
            )
            new_messages.append(agent_message)

        input_message_id = None
        if a2a_task.status and a2a_task.status.message:
            input_message_id = a2a_task.status.message.message_id
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=response_text,
                turn_number=turn_number,
                context_id=a2a_task.context_id,
                message_id=input_message_id,
                is_multiturn=is_multiturn,
                status="input_required",
                final=is_final,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

        return TaskStateResult(
            status=TaskState.input_required,
            error=response_text,
            history=new_messages,
            agent_card=agent_card.model_dump(exclude_none=True),
        )

    if a2a_task.status.state in {TaskState.failed, TaskState.rejected}:
        error_msg = extract_error_message(a2a_task, "Task failed without error message")
        if a2a_task.history:
            new_messages.extend(a2a_task.history)
        return TaskStateResult(
            status=TaskState.failed,
            error=error_msg,
            history=new_messages,
        )

    if a2a_task.status.state == TaskState.auth_required:
        error_msg = extract_error_message(a2a_task, "Authentication required")
        return TaskStateResult(
            status=TaskState.auth_required,
            error=error_msg,
            history=new_messages,
        )

    if a2a_task.status.state == TaskState.canceled:
        error_msg = extract_error_message(a2a_task, "Task was canceled")
        return TaskStateResult(
            status=TaskState.canceled,
            error=error_msg,
            history=new_messages,
        )

    if a2a_task.status.state in PENDING_STATES:
        return None

    return None


async def send_message_and_get_task_id(
    event_stream: AsyncIterator[SendMessageEvent],
    new_messages: list[Message],
    agent_card: AgentCard,
    turn_number: int,
    is_multiturn: bool,
    agent_role: str | None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
    context_id: str | None = None,
) -> str | TaskStateResult:
    """Send message and process initial response.

    Handles the common pattern of sending a message and either:
    - Getting an immediate Message response (task completed synchronously)
    - Getting a Task that needs polling/waiting for completion

    Args:
        event_stream: Async iterator from client.send_message()
        new_messages: List to collect messages (modified in place)
        agent_card: The agent card
        turn_number: Current turn number
        is_multiturn: Whether multi-turn conversation
        agent_role: Agent role for logging
        from_task: Optional CrewAI Task object for event metadata.
        from_agent: Optional CrewAI Agent object for event metadata.
        endpoint: Optional A2A endpoint URL.
        a2a_agent_name: Optional A2A agent name.
        context_id: Optional A2A context ID for correlation.

    Returns:
        Task ID string if agent needs polling/waiting, or TaskStateResult if done.
    """
    try:
        async for event in event_stream:
            if isinstance(event, Message):
                new_messages.append(event)
                result_parts = [
                    part.root.text for part in event.parts if part.root.kind == "text"
                ]
                response_text = " ".join(result_parts) if result_parts else ""

                crewai_event_bus.emit(
                    None,
                    A2AResponseReceivedEvent(
                        response=response_text,
                        turn_number=turn_number,
                        context_id=event.context_id,
                        message_id=event.message_id,
                        is_multiturn=is_multiturn,
                        status="completed",
                        final=True,
                        agent_role=agent_role,
                        endpoint=endpoint,
                        a2a_agent_name=a2a_agent_name,
                        from_task=from_task,
                        from_agent=from_agent,
                    ),
                )

                return TaskStateResult(
                    status=TaskState.completed,
                    result=response_text,
                    history=new_messages,
                    agent_card=agent_card.model_dump(exclude_none=True),
                )

            if isinstance(event, tuple):
                a2a_task, _ = event

                if a2a_task.status.state in TERMINAL_STATES | ACTIONABLE_STATES:
                    result = process_task_state(
                        a2a_task=a2a_task,
                        new_messages=new_messages,
                        agent_card=agent_card,
                        turn_number=turn_number,
                        is_multiturn=is_multiturn,
                        agent_role=agent_role,
                        endpoint=endpoint,
                        a2a_agent_name=a2a_agent_name,
                        from_task=from_task,
                        from_agent=from_agent,
                    )
                    if result:
                        return result

                return a2a_task.id

        return TaskStateResult(
            status=TaskState.failed,
            error="No task ID received from initial message",
            history=new_messages,
        )

    except A2AClientHTTPError as e:
        error_msg = f"HTTP Error {e.status_code}: {e!s}"

        error_message = Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=error_msg))],
            context_id=context_id,
        )
        new_messages.append(error_message)

        crewai_event_bus.emit(
            None,
            A2AConnectionErrorEvent(
                endpoint=endpoint or "",
                error=str(e),
                error_type="http_error",
                status_code=e.status_code,
                a2a_agent_name=a2a_agent_name,
                operation="send_message",
                context_id=context_id,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=error_msg,
                turn_number=turn_number,
                context_id=context_id,
                is_multiturn=is_multiturn,
                status="failed",
                final=True,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        return TaskStateResult(
            status=TaskState.failed,
            error=error_msg,
            history=new_messages,
        )

    except Exception as e:
        error_msg = f"Unexpected error during send_message: {e!s}"

        error_message = Message(
            role=Role.agent,
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=error_msg))],
            context_id=context_id,
        )
        new_messages.append(error_message)

        crewai_event_bus.emit(
            None,
            A2AConnectionErrorEvent(
                endpoint=endpoint or "",
                error=str(e),
                error_type="unexpected_error",
                a2a_agent_name=a2a_agent_name,
                operation="send_message",
                context_id=context_id,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        crewai_event_bus.emit(
            None,
            A2AResponseReceivedEvent(
                response=error_msg,
                turn_number=turn_number,
                context_id=context_id,
                is_multiturn=is_multiturn,
                status="failed",
                final=True,
                agent_role=agent_role,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )
        return TaskStateResult(
            status=TaskState.failed,
            error=error_msg,
            history=new_messages,
        )

    finally:
        aclose = getattr(event_stream, "aclose", None)
        if aclose:
            await aclose()
