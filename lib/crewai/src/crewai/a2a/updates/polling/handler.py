"""Polling update mechanism handler."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from a2a.client import Client
from a2a.types import (
    AgentCard,
    Message,
    TaskQueryParams,
    TaskState,
)

from crewai.a2a.errors import A2APollingTimeoutError
from crewai.a2a.task_helpers import TaskStateResult, process_task_state
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2APollingStartedEvent,
    A2APollingStatusEvent,
    A2AResponseReceivedEvent,
)


if TYPE_CHECKING:
    from a2a.types import Task as A2ATask


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.failed,
    TaskState.rejected,
    TaskState.canceled,
}


async def poll_task_until_complete(
    client: Client,
    task_id: str,
    polling_interval: float,
    polling_timeout: float,
    agent_branch: Any | None = None,
    history_length: int = 100,
    max_polls: int | None = None,
) -> A2ATask:
    """Poll task status until terminal state reached.

    Args:
        client: A2A client instance
        task_id: Task ID to poll
        polling_interval: Seconds between poll attempts
        polling_timeout: Max seconds before timeout
        agent_branch: Agent tree branch for logging
        history_length: Number of messages to retrieve per poll
        max_polls: Max number of poll attempts (None = unlimited)

    Returns:
        Final task object in terminal state

    Raises:
        A2APollingTimeoutError: If polling exceeds timeout or max_polls
    """
    start_time = time.monotonic()
    poll_count = 0

    while True:
        poll_count += 1
        task = await client.get_task(
            TaskQueryParams(id=task_id, history_length=history_length)
        )

        elapsed = time.monotonic() - start_time
        crewai_event_bus.emit(
            agent_branch,
            A2APollingStatusEvent(
                task_id=task_id,
                state=str(task.status.state.value) if task.status.state else "unknown",
                elapsed_seconds=elapsed,
                poll_count=poll_count,
            ),
        )

        if task.status.state in TERMINAL_STATES:
            return task

        if task.status.state in {TaskState.input_required, TaskState.auth_required}:
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


async def execute_polling_delegation(
    client: Client,
    message: Message,
    polling_interval: float,
    polling_timeout: float,
    endpoint: str,
    agent_branch: Any | None,
    turn_number: int,
    is_multiturn: bool,
    agent_role: str | None,
    new_messages: list[Message],
    agent_card: AgentCard,
    history_length: int = 100,
    max_polls: int | None = None,
) -> TaskStateResult:
    """Execute A2A delegation using polling for updates.

    Args:
        client: A2A client instance
        message: Message to send
        polling_interval: Seconds between poll attempts
        polling_timeout: Max seconds before timeout
        endpoint: A2A agent endpoint URL
        agent_branch: Agent tree branch for logging
        turn_number: Current turn number
        is_multiturn: Whether this is a multi-turn conversation
        agent_role: Agent role for logging
        new_messages: List to collect messages
        agent_card: The agent card
        history_length: Number of messages to retrieve per poll
        max_polls: Max number of poll attempts (None = unlimited)

    Returns:
        Dictionary with status, result/error, and history
    """
    task_id: str | None = None

    async for event in client.send_message(message):
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
                    is_multiturn=is_multiturn,
                    status="completed",
                    agent_role=agent_role,
                ),
            )

            return TaskStateResult(
                status=TaskState.completed,
                result=response_text,
                history=new_messages,
                agent_card=agent_card,
            )

        if isinstance(event, tuple):
            a2a_task, _ = event
            task_id = a2a_task.id

            if a2a_task.status.state in TERMINAL_STATES | {
                TaskState.input_required,
                TaskState.auth_required,
            }:
                result = process_task_state(
                    a2a_task=a2a_task,
                    new_messages=new_messages,
                    agent_card=agent_card,
                    turn_number=turn_number,
                    is_multiturn=is_multiturn,
                    agent_role=agent_role,
                )
                if result:
                    return result
            break

    if not task_id:
        return TaskStateResult(
            status=TaskState.failed,
            error="No task ID received from initial message",
            history=new_messages,
        )

    crewai_event_bus.emit(
        agent_branch,
        A2APollingStartedEvent(
            task_id=task_id,
            polling_interval=polling_interval,
            endpoint=endpoint,
        ),
    )

    final_task = await poll_task_until_complete(
        client=client,
        task_id=task_id,
        polling_interval=polling_interval,
        polling_timeout=polling_timeout,
        agent_branch=agent_branch,
        history_length=history_length,
        max_polls=max_polls,
    )

    result = process_task_state(
        a2a_task=final_task,
        new_messages=new_messages,
        agent_card=agent_card,
        turn_number=turn_number,
        is_multiturn=is_multiturn,
        agent_role=agent_role,
    )
    if result:
        return result

    return TaskStateResult(
        status=TaskState.failed,
        error=f"Unexpected task state: {final_task.status.state}",
        history=new_messages,
    )
