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
from crewai.events.types.a2a_events import (
    A2AArtifactReceivedEvent,
    A2AConnectionErrorEvent,
    A2AResponseReceivedEvent,
    A2AStreamingChunkEvent,
    A2AStreamingStartedEvent,
)


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
        endpoint = kwargs.get("endpoint")
        a2a_agent_name = kwargs.get("a2a_agent_name")
        from_task = kwargs.get("from_task")
        from_agent = kwargs.get("from_agent")
        agent_branch = kwargs.get("agent_branch")

        result_parts: list[str] = []
        final_result: TaskStateResult | None = None
        event_stream = client.send_message(message)
        chunk_index = 0

        crewai_event_bus.emit(
            agent_branch,
            A2AStreamingStartedEvent(
                task_id=task_id,
                context_id=context_id,
                endpoint=endpoint or "",
                a2a_agent_name=a2a_agent_name,
                turn_number=turn_number,
                is_multiturn=is_multiturn,
                agent_role=agent_role,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

        try:
            async for event in event_stream:
                if isinstance(event, Message):
                    new_messages.append(event)
                    message_context_id = event.context_id or context_id
                    for part in event.parts:
                        if part.root.kind == "text":
                            text = part.root.text
                            result_parts.append(text)
                            crewai_event_bus.emit(
                                agent_branch,
                                A2AStreamingChunkEvent(
                                    task_id=event.task_id or task_id,
                                    context_id=message_context_id,
                                    chunk=text,
                                    chunk_index=chunk_index,
                                    endpoint=endpoint,
                                    a2a_agent_name=a2a_agent_name,
                                    turn_number=turn_number,
                                    is_multiturn=is_multiturn,
                                    from_task=from_task,
                                    from_agent=from_agent,
                                ),
                            )
                            chunk_index += 1

                elif isinstance(event, tuple):
                    a2a_task, update = event

                    if isinstance(update, TaskArtifactUpdateEvent):
                        artifact = update.artifact
                        result_parts.extend(
                            part.root.text
                            for part in artifact.parts
                            if part.root.kind == "text"
                        )
                        artifact_size = None
                        if artifact.parts:
                            artifact_size = sum(
                                len(p.root.text.encode("utf-8"))
                                if p.root.kind == "text"
                                else len(getattr(p.root, "data", b""))
                                for p in artifact.parts
                            )
                        effective_context_id = a2a_task.context_id or context_id
                        crewai_event_bus.emit(
                            agent_branch,
                            A2AArtifactReceivedEvent(
                                task_id=a2a_task.id,
                                artifact_id=artifact.artifact_id,
                                artifact_name=artifact.name,
                                artifact_description=artifact.description,
                                mime_type=artifact.parts[0].root.kind
                                if artifact.parts
                                else None,
                                size_bytes=artifact_size,
                                append=update.append or False,
                                last_chunk=update.last_chunk or False,
                                endpoint=endpoint,
                                a2a_agent_name=a2a_agent_name,
                                context_id=effective_context_id,
                                turn_number=turn_number,
                                is_multiturn=is_multiturn,
                                from_task=from_task,
                                from_agent=from_agent,
                            ),
                        )

                    is_final_update = False
                    if isinstance(update, TaskStatusUpdateEvent):
                        is_final_update = update.final
                        if (
                            update.status
                            and update.status.message
                            and update.status.message.parts
                        ):
                            result_parts.extend(
                                part.root.text
                                for part in update.status.message.parts
                                if part.root.kind == "text" and part.root.text
                            )

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
                        endpoint=endpoint,
                        a2a_agent_name=a2a_agent_name,
                        from_task=from_task,
                        from_agent=from_agent,
                        is_final=is_final_update,
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
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=endpoint or "",
                    error=str(e),
                    error_type="http_error",
                    status_code=e.status_code,
                    a2a_agent_name=a2a_agent_name,
                    operation="streaming",
                    context_id=context_id,
                    task_id=task_id,
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )
            crewai_event_bus.emit(
                agent_branch,
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
            error_msg = f"Unexpected error during streaming: {e!s}"

            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=error_msg))],
                context_id=context_id,
                task_id=task_id,
            )
            new_messages.append(error_message)

            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=endpoint or "",
                    error=str(e),
                    error_type="unexpected_error",
                    a2a_agent_name=a2a_agent_name,
                    operation="streaming",
                    context_id=context_id,
                    task_id=task_id,
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )
            crewai_event_bus.emit(
                agent_branch,
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
                try:
                    await aclose()
                except Exception as close_error:
                    crewai_event_bus.emit(
                        agent_branch,
                        A2AConnectionErrorEvent(
                            endpoint=endpoint or "",
                            error=str(close_error),
                            error_type="stream_close_error",
                            a2a_agent_name=a2a_agent_name,
                            operation="stream_close",
                            context_id=context_id,
                            task_id=task_id,
                            from_task=from_task,
                            from_agent=from_agent,
                        ),
                    )

        if final_result:
            return final_result

        return TaskStateResult(
            status=TaskState.completed,
            result=" ".join(result_parts) if result_parts else "",
            history=new_messages,
            agent_card=agent_card.model_dump(exclude_none=True),
        )
