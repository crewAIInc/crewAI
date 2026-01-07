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
from typing_extensions import Unpack

from crewai.a2a.task_helpers import (
    ACTIONABLE_STATES,
    TERMINAL_STATES,
    TaskStateResult,
    process_task_state,
)
from crewai.a2a.updates.base import StreamingHandlerKwargs
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import A2AResponseReceivedEvent


class StreamingHandler:
    """SSE streaming-based update handler."""

    @staticmethod
    async def execute(
        client: Client,
        message: Message,
        new_messages: list[Message],
        agent_card: AgentCard,
        **kwargs: Unpack[StreamingHandlerKwargs],
    ) -> TaskStateResult:
        """Execute A2A delegation using SSE streaming for updates.

        Args:
            client: A2A client instance.
            message: Message to send.
            new_messages: List to collect messages.
            agent_card: The agent card.
            **kwargs: Streaming-specific parameters.

        Returns:
            Dictionary with status, result/error, and history.
        """
        context_id = kwargs.get("context_id")
        task_id = kwargs.get("task_id")
        turn_number = kwargs.get("turn_number", 0)
        is_multiturn = kwargs.get("is_multiturn", False)
        agent_role = kwargs.get("agent_role")

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

                    if (
                        not is_final_update
                        and a2a_task.status.state
                        not in TERMINAL_STATES | ACTIONABLE_STATES
                    ):
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

        finally:
            aclose = getattr(event_stream, "aclose", None)
            if aclose:
                await aclose()

        if final_result:
            return final_result

        return TaskStateResult(
            status=TaskState.completed,
            result=" ".join(result_parts) if result_parts else "",
            history=new_messages,
            agent_card=agent_card,
        )
