"""Polling update mechanism handler."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
import uuid

from a2a.client import Client
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    TaskQueryParams,
    TaskState,
    TextPart,
)
from typing_extensions import Unpack

from crewai.a2a.errors import A2APollingTimeoutError
from crewai.a2a.task_helpers import (
    ACTIONABLE_STATES,
    TERMINAL_STATES,
    TaskStateResult,
    process_task_state,
    send_message_and_get_task_id,
)
from crewai.a2a.updates.base import PollingHandlerKwargs
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConnectionErrorEvent,
    A2APollingStartedEvent,
    A2APollingStatusEvent,
    A2AResponseReceivedEvent,
)


if TYPE_CHECKING:
    from a2a.types import Task as A2ATask


async def _poll_task_until_complete(
    client: Client,
    task_id: str,
    polling_interval: float,
    polling_timeout: float,
    agent_branch: Any | None = None,
    history_length: int = 100,
    max_polls: int | None = None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    context_id: str | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
) -> A2ATask:
    """Poll task status until terminal state reached.

    Args:
        client: A2A client instance.
        task_id: Task ID to poll.
        polling_interval: Seconds between poll attempts.
        polling_timeout: Max seconds before timeout.
        agent_branch: Agent tree branch for logging.
        history_length: Number of messages to retrieve per poll.
        max_polls: Max number of poll attempts (None = unlimited).
        from_task: Optional CrewAI Task object for event metadata.
        from_agent: Optional CrewAI Agent object for event metadata.
        context_id: A2A context ID for correlation.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent from agent card.

    Returns:
        Final task object in terminal state.

    Raises:
        A2APollingTimeoutError: If polling exceeds timeout or max_polls.
    """
    start_time = time.monotonic()
    poll_count = 0

    while True:
        poll_count += 1
        task = await client.get_task(
            TaskQueryParams(id=task_id, history_length=history_length)
        )

        elapsed = time.monotonic() - start_time
        effective_context_id = task.context_id or context_id
        crewai_event_bus.emit(
            agent_branch,
            A2APollingStatusEvent(
                task_id=task_id,
                context_id=effective_context_id,
                state=str(task.status.state.value),
                elapsed_seconds=elapsed,
                poll_count=poll_count,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

        if task.status.state in TERMINAL_STATES | ACTIONABLE_STATES:
            return task

        if elapsed > polling_timeout:
            raise A2APollingTimeoutError(
                f"Polling timeout after {polling_timeout}s ({poll_count} polls)"
            )

        if max_polls and poll_count >= max_polls:
            raise A2APollingTimeoutError(
                f"Max polls ({max_polls}) exceeded after {elapsed:.1f}s"
            )

        await asyncio.sleep(polling_interval)


class PollingHandler:
    """Polling-based update handler."""

    @staticmethod
    async def execute(
        client: Client,
        message: Message,
        new_messages: list[Message],
        agent_card: AgentCard,
        **kwargs: Unpack[PollingHandlerKwargs],
    ) -> TaskStateResult:
        """Execute A2A delegation using polling for updates.

        Args:
            client: A2A client instance.
            message: Message to send.
            new_messages: List to collect messages.
            agent_card: The agent card.
            **kwargs: Polling-specific parameters.

        Returns:
            Dictionary with status, result/error, and history.
        """
        polling_interval = kwargs.get("polling_interval", 2.0)
        polling_timeout = kwargs.get("polling_timeout", 300.0)
        endpoint = kwargs.get("endpoint", "")
        agent_branch = kwargs.get("agent_branch")
        turn_number = kwargs.get("turn_number", 0)
        is_multiturn = kwargs.get("is_multiturn", False)
        agent_role = kwargs.get("agent_role")
        history_length = kwargs.get("history_length", 100)
        max_polls = kwargs.get("max_polls")
        context_id = kwargs.get("context_id")
        task_id = kwargs.get("task_id")
        a2a_agent_name = kwargs.get("a2a_agent_name")
        from_task = kwargs.get("from_task")
        from_agent = kwargs.get("from_agent")

        try:
            result_or_task_id = await send_message_and_get_task_id(
                event_stream=client.send_message(message),
                new_messages=new_messages,
                agent_card=agent_card,
                turn_number=turn_number,
                is_multiturn=is_multiturn,
                agent_role=agent_role,
                from_task=from_task,
                from_agent=from_agent,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                context_id=context_id,
            )

            if not isinstance(result_or_task_id, str):
                return result_or_task_id

            task_id = result_or_task_id

            crewai_event_bus.emit(
                agent_branch,
                A2APollingStartedEvent(
                    task_id=task_id,
                    context_id=context_id,
                    polling_interval=polling_interval,
                    endpoint=endpoint,
                    a2a_agent_name=a2a_agent_name,
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )

            final_task = await _poll_task_until_complete(
                client=client,
                task_id=task_id,
                polling_interval=polling_interval,
                polling_timeout=polling_timeout,
                agent_branch=agent_branch,
                history_length=history_length,
                max_polls=max_polls,
                from_task=from_task,
                from_agent=from_agent,
                context_id=context_id,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
            )

            result = process_task_state(
                a2a_task=final_task,
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

            return TaskStateResult(
                status=TaskState.failed,
                error=f"Unexpected task state: {final_task.status.state}",
                history=new_messages,
            )

        except A2APollingTimeoutError as e:
            error_msg = str(e)

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
                    endpoint=endpoint,
                    error=str(e),
                    error_type="http_error",
                    status_code=e.status_code,
                    a2a_agent_name=a2a_agent_name,
                    operation="polling",
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
            error_msg = f"Unexpected error during polling: {e!s}"

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
                    endpoint=endpoint,
                    error=str(e),
                    error_type="unexpected_error",
                    a2a_agent_name=a2a_agent_name,
                    operation="polling",
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
