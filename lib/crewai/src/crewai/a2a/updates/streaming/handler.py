"""Streaming (SSE) update mechanism handler."""

from __future__ import annotations

import asyncio
import logging
from typing import Final
import uuid

from a2a.client import Client
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskQueryParams,
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
from crewai.a2a.updates.base import StreamingHandlerKwargs, extract_common_params
from crewai.a2a.updates.streaming.params import (
    process_status_update,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AArtifactReceivedEvent,
    A2AConnectionErrorEvent,
    A2AResponseReceivedEvent,
    A2AStreamingChunkEvent,
    A2AStreamingStartedEvent,
)


logger = logging.getLogger(__name__)

MAX_RESUBSCRIBE_ATTEMPTS: Final[int] = 3
RESUBSCRIBE_BACKOFF_BASE: Final[float] = 1.0


class StreamingHandler:
    """SSE streaming-based update handler."""

    @staticmethod
    async def _try_recover_from_interruption(  # type: ignore[misc]
        client: Client,
        task_id: str,
        new_messages: list[Message],
        agent_card: AgentCard,
        result_parts: list[str],
        **kwargs: Unpack[StreamingHandlerKwargs],
    ) -> TaskStateResult | None:
        """Attempt to recover from a stream interruption by checking task state.

        If the task completed while we were disconnected, returns the result.
        If the task is still running, attempts to resubscribe and continue.

        Args:
            client: A2A client instance.
            task_id: The task ID to recover.
            new_messages: List of collected messages.
            agent_card: The agent card.
            result_parts: Accumulated result text parts.
            **kwargs: Handler parameters.

        Returns:
            TaskStateResult if recovery succeeded (task finished or resubscribe worked).
            None if recovery not possible (caller should handle failure).

        Note:
            When None is returned, recovery failed and the original exception should
            be handled by the caller. All recovery attempts are logged.
        """
        params = extract_common_params(kwargs)  # type: ignore[arg-type]

        try:
            a2a_task: Task = await client.get_task(TaskQueryParams(id=task_id))

            if a2a_task.status.state in TERMINAL_STATES:
                logger.info(
                    "Task completed during stream interruption",
                    extra={"task_id": task_id, "state": str(a2a_task.status.state)},
                )
                return process_task_state(
                    a2a_task=a2a_task,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    turn_number=params.turn_number,
                    is_multiturn=params.is_multiturn,
                    agent_role=params.agent_role,
                    result_parts=result_parts,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                )

            if a2a_task.status.state in ACTIONABLE_STATES:
                logger.info(
                    "Task in actionable state during stream interruption",
                    extra={"task_id": task_id, "state": str(a2a_task.status.state)},
                )
                return process_task_state(
                    a2a_task=a2a_task,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    turn_number=params.turn_number,
                    is_multiturn=params.is_multiturn,
                    agent_role=params.agent_role,
                    result_parts=result_parts,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                    is_final=False,
                )

            logger.info(
                "Task still running, attempting resubscribe",
                extra={"task_id": task_id, "state": str(a2a_task.status.state)},
            )

            for attempt in range(MAX_RESUBSCRIBE_ATTEMPTS):
                try:
                    backoff = RESUBSCRIBE_BACKOFF_BASE * (2**attempt)
                    if attempt > 0:
                        await asyncio.sleep(backoff)

                    event_stream = client.resubscribe(TaskIdParams(id=task_id))

                    async for event in event_stream:
                        if isinstance(event, tuple):
                            resubscribed_task, update = event

                            is_final_update = (
                                process_status_update(update, result_parts)
                                if isinstance(update, TaskStatusUpdateEvent)
                                else False
                            )

                            if isinstance(update, TaskArtifactUpdateEvent):
                                artifact = update.artifact
                                result_parts.extend(
                                    part.root.text
                                    for part in artifact.parts
                                    if part.root.kind == "text"
                                )

                            if (
                                is_final_update
                                or resubscribed_task.status.state
                                in TERMINAL_STATES | ACTIONABLE_STATES
                            ):
                                return process_task_state(
                                    a2a_task=resubscribed_task,
                                    new_messages=new_messages,
                                    agent_card=agent_card,
                                    turn_number=params.turn_number,
                                    is_multiturn=params.is_multiturn,
                                    agent_role=params.agent_role,
                                    result_parts=result_parts,
                                    endpoint=params.endpoint,
                                    a2a_agent_name=params.a2a_agent_name,
                                    from_task=params.from_task,
                                    from_agent=params.from_agent,
                                    is_final=is_final_update,
                                )

                        elif isinstance(event, Message):
                            new_messages.append(event)
                            result_parts.extend(
                                part.root.text
                                for part in event.parts
                                if part.root.kind == "text"
                            )

                    final_task = await client.get_task(TaskQueryParams(id=task_id))
                    return process_task_state(
                        a2a_task=final_task,
                        new_messages=new_messages,
                        agent_card=agent_card,
                        turn_number=params.turn_number,
                        is_multiturn=params.is_multiturn,
                        agent_role=params.agent_role,
                        result_parts=result_parts,
                        endpoint=params.endpoint,
                        a2a_agent_name=params.a2a_agent_name,
                        from_task=params.from_task,
                        from_agent=params.from_agent,
                    )

                except Exception as resubscribe_error:  # noqa: PERF203
                    logger.warning(
                        "Resubscribe attempt failed",
                        extra={
                            "task_id": task_id,
                            "attempt": attempt + 1,
                            "max_attempts": MAX_RESUBSCRIBE_ATTEMPTS,
                            "error": str(resubscribe_error),
                        },
                    )
                    if attempt == MAX_RESUBSCRIBE_ATTEMPTS - 1:
                        return None

        except Exception as e:
            logger.warning(
                "Failed to recover from stream interruption due to unexpected error",
                extra={
                    "task_id": task_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return None

        logger.warning(
            "Recovery exhausted all resubscribe attempts without success",
            extra={"task_id": task_id, "max_attempts": MAX_RESUBSCRIBE_ATTEMPTS},
        )
        return None

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
        task_id = kwargs.get("task_id")
        agent_branch = kwargs.get("agent_branch")
        params = extract_common_params(kwargs)

        result_parts: list[str] = []
        final_result: TaskStateResult | None = None
        event_stream = client.send_message(message)
        chunk_index = 0
        current_task_id: str | None = task_id

        crewai_event_bus.emit(
            agent_branch,
            A2AStreamingStartedEvent(
                task_id=task_id,
                context_id=params.context_id,
                endpoint=params.endpoint,
                a2a_agent_name=params.a2a_agent_name,
                turn_number=params.turn_number,
                is_multiturn=params.is_multiturn,
                agent_role=params.agent_role,
                from_task=params.from_task,
                from_agent=params.from_agent,
            ),
        )

        try:
            async for event in event_stream:
                if isinstance(event, tuple):
                    a2a_task, _ = event
                    current_task_id = a2a_task.id

                if isinstance(event, Message):
                    new_messages.append(event)
                    message_context_id = event.context_id or params.context_id
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
                                    endpoint=params.endpoint,
                                    a2a_agent_name=params.a2a_agent_name,
                                    turn_number=params.turn_number,
                                    is_multiturn=params.is_multiturn,
                                    from_task=params.from_task,
                                    from_agent=params.from_agent,
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
                                len(p.root.text.encode())
                                if p.root.kind == "text"
                                else len(getattr(p.root, "data", b""))
                                for p in artifact.parts
                            )
                        effective_context_id = a2a_task.context_id or params.context_id
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
                                endpoint=params.endpoint,
                                a2a_agent_name=params.a2a_agent_name,
                                context_id=effective_context_id,
                                turn_number=params.turn_number,
                                is_multiturn=params.is_multiturn,
                                from_task=params.from_task,
                                from_agent=params.from_agent,
                            ),
                        )

                    is_final_update = (
                        process_status_update(update, result_parts)
                        if isinstance(update, TaskStatusUpdateEvent)
                        else False
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
                        turn_number=params.turn_number,
                        is_multiturn=params.is_multiturn,
                        agent_role=params.agent_role,
                        result_parts=result_parts,
                        endpoint=params.endpoint,
                        a2a_agent_name=params.a2a_agent_name,
                        from_task=params.from_task,
                        from_agent=params.from_agent,
                        is_final=is_final_update,
                    )
                    if final_result:
                        break

        except A2AClientHTTPError as e:
            if current_task_id:
                logger.info(
                    "Stream interrupted with HTTP error, attempting recovery",
                    extra={
                        "task_id": current_task_id,
                        "error": str(e),
                        "status_code": e.status_code,
                    },
                )
                recovery_kwargs = {k: v for k, v in kwargs.items() if k != "task_id"}
                recovered_result = (
                    await StreamingHandler._try_recover_from_interruption(
                        client=client,
                        task_id=current_task_id,
                        new_messages=new_messages,
                        agent_card=agent_card,
                        result_parts=result_parts,
                        **recovery_kwargs,
                    )
                )
                if recovered_result:
                    logger.info(
                        "Successfully recovered task after HTTP error",
                        extra={
                            "task_id": current_task_id,
                            "status": str(recovered_result.get("status")),
                        },
                    )
                    return recovered_result

                logger.warning(
                    "Failed to recover from HTTP error, returning failure",
                    extra={
                        "task_id": current_task_id,
                        "status_code": e.status_code,
                        "original_error": str(e),
                    },
                )

            error_msg = f"HTTP Error {e.status_code}: {e!s}"
            error_type = "http_error"
            status_code = e.status_code

            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=error_msg))],
                context_id=params.context_id,
                task_id=task_id,
            )
            new_messages.append(error_message)

            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=params.endpoint,
                    error=str(e),
                    error_type=error_type,
                    status_code=status_code,
                    a2a_agent_name=params.a2a_agent_name,
                    operation="streaming",
                    context_id=params.context_id,
                    task_id=task_id,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AResponseReceivedEvent(
                    response=error_msg,
                    turn_number=params.turn_number,
                    context_id=params.context_id,
                    is_multiturn=params.is_multiturn,
                    status="failed",
                    final=True,
                    agent_role=params.agent_role,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            return TaskStateResult(
                status=TaskState.failed,
                error=error_msg,
                history=new_messages,
            )

        except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError) as e:
            error_type = type(e).__name__.lower()
            if current_task_id:
                logger.info(
                    f"Stream interrupted with {error_type}, attempting recovery",
                    extra={"task_id": current_task_id, "error": str(e)},
                )
                recovery_kwargs = {k: v for k, v in kwargs.items() if k != "task_id"}
                recovered_result = (
                    await StreamingHandler._try_recover_from_interruption(
                        client=client,
                        task_id=current_task_id,
                        new_messages=new_messages,
                        agent_card=agent_card,
                        result_parts=result_parts,
                        **recovery_kwargs,
                    )
                )
                if recovered_result:
                    logger.info(
                        f"Successfully recovered task after {error_type}",
                        extra={
                            "task_id": current_task_id,
                            "status": str(recovered_result.get("status")),
                        },
                    )
                    return recovered_result

                logger.warning(
                    f"Failed to recover from {error_type}, returning failure",
                    extra={
                        "task_id": current_task_id,
                        "error_type": error_type,
                        "original_error": str(e),
                    },
                )

            error_msg = f"Connection error during streaming: {e!s}"
            status_code = None

            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=error_msg))],
                context_id=params.context_id,
                task_id=task_id,
            )
            new_messages.append(error_message)

            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=params.endpoint,
                    error=str(e),
                    error_type=error_type,
                    status_code=status_code,
                    a2a_agent_name=params.a2a_agent_name,
                    operation="streaming",
                    context_id=params.context_id,
                    task_id=task_id,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AResponseReceivedEvent(
                    response=error_msg,
                    turn_number=params.turn_number,
                    context_id=params.context_id,
                    is_multiturn=params.is_multiturn,
                    status="failed",
                    final=True,
                    agent_role=params.agent_role,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            return TaskStateResult(
                status=TaskState.failed,
                error=error_msg,
                history=new_messages,
            )

        except Exception as e:
            logger.exception(
                "Unexpected error during streaming",
                extra={
                    "task_id": current_task_id,
                    "error_type": type(e).__name__,
                    "endpoint": params.endpoint,
                },
            )
            error_msg = f"Unexpected error during streaming: {type(e).__name__}: {e!s}"
            error_type = "unexpected_error"
            status_code = None

            error_message = Message(
                role=Role.agent,
                message_id=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=error_msg))],
                context_id=params.context_id,
                task_id=task_id,
            )
            new_messages.append(error_message)

            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=params.endpoint,
                    error=str(e),
                    error_type=error_type,
                    status_code=status_code,
                    a2a_agent_name=params.a2a_agent_name,
                    operation="streaming",
                    context_id=params.context_id,
                    task_id=task_id,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
                ),
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AResponseReceivedEvent(
                    response=error_msg,
                    turn_number=params.turn_number,
                    context_id=params.context_id,
                    is_multiturn=params.is_multiturn,
                    status="failed",
                    final=True,
                    agent_role=params.agent_role,
                    endpoint=params.endpoint,
                    a2a_agent_name=params.a2a_agent_name,
                    from_task=params.from_task,
                    from_agent=params.from_agent,
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
                            endpoint=params.endpoint,
                            error=str(close_error),
                            error_type="stream_close_error",
                            a2a_agent_name=params.a2a_agent_name,
                            operation="stream_close",
                            context_id=params.context_id,
                            task_id=task_id,
                            from_task=params.from_task,
                            from_agent=params.from_agent,
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
