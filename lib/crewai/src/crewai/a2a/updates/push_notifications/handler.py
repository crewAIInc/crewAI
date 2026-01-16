"""Push notification (webhook) update mechanism handler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
import uuid

from a2a.client import Client
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    TaskState,
    TextPart,
)
from typing_extensions import Unpack

from crewai.a2a.task_helpers import (
    TaskStateResult,
    process_task_state,
    send_message_and_get_task_id,
)
from crewai.a2a.updates.base import (
    PushNotificationHandlerKwargs,
    PushNotificationResultStore,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AConnectionErrorEvent,
    A2APushNotificationRegisteredEvent,
    A2APushNotificationTimeoutEvent,
    A2AResponseReceivedEvent,
)


if TYPE_CHECKING:
    from a2a.types import Task as A2ATask


logger = logging.getLogger(__name__)


async def _wait_for_push_result(
    task_id: str,
    result_store: PushNotificationResultStore,
    timeout: float,
    poll_interval: float,
    agent_branch: Any | None = None,
    from_task: Any | None = None,
    from_agent: Any | None = None,
    context_id: str | None = None,
    endpoint: str | None = None,
    a2a_agent_name: str | None = None,
) -> A2ATask | None:
    """Wait for push notification result.

    Args:
        task_id: Task ID to wait for.
        result_store: Store to retrieve results from.
        timeout: Max seconds to wait.
        poll_interval: Seconds between polling attempts.
        agent_branch: Agent tree branch for logging.
        from_task: Optional CrewAI Task object for event metadata.
        from_agent: Optional CrewAI Agent object for event metadata.
        context_id: A2A context ID for correlation.
        endpoint: A2A agent endpoint URL.
        a2a_agent_name: Name of the A2A agent.

    Returns:
        Final task object, or None if timeout.
    """
    task = await result_store.wait_for_result(
        task_id=task_id,
        timeout=timeout,
        poll_interval=poll_interval,
    )

    if task is None:
        crewai_event_bus.emit(
            agent_branch,
            A2APushNotificationTimeoutEvent(
                task_id=task_id,
                context_id=context_id,
                timeout_seconds=timeout,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
                from_task=from_task,
                from_agent=from_agent,
            ),
        )

    return task


class PushNotificationHandler:
    """Push notification (webhook) based update handler."""

    @staticmethod
    async def execute(
        client: Client,
        message: Message,
        new_messages: list[Message],
        agent_card: AgentCard,
        **kwargs: Unpack[PushNotificationHandlerKwargs],
    ) -> TaskStateResult:
        """Execute A2A delegation using push notifications for updates.

        Args:
            client: A2A client instance.
            message: Message to send.
            new_messages: List to collect messages.
            agent_card: The agent card.
            **kwargs: Push notification-specific parameters.

        Returns:
            Dictionary with status, result/error, and history.

        Raises:
            ValueError: If result_store or config not provided.
        """
        config = kwargs.get("config")
        result_store = kwargs.get("result_store")
        polling_timeout = kwargs.get("polling_timeout", 300.0)
        polling_interval = kwargs.get("polling_interval", 2.0)
        agent_branch = kwargs.get("agent_branch")
        turn_number = kwargs.get("turn_number", 0)
        is_multiturn = kwargs.get("is_multiturn", False)
        agent_role = kwargs.get("agent_role")
        context_id = kwargs.get("context_id")
        task_id = kwargs.get("task_id")
        endpoint = kwargs.get("endpoint")
        a2a_agent_name = kwargs.get("a2a_agent_name")
        from_task = kwargs.get("from_task")
        from_agent = kwargs.get("from_agent")

        if config is None:
            error_msg = (
                "PushNotificationConfig is required for push notification handler"
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=endpoint or "",
                    error=error_msg,
                    error_type="configuration_error",
                    a2a_agent_name=a2a_agent_name,
                    operation="push_notification",
                    context_id=context_id,
                    task_id=task_id,
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )
            return TaskStateResult(
                status=TaskState.failed,
                error=error_msg,
                history=new_messages,
            )

        if result_store is None:
            error_msg = (
                "PushNotificationResultStore is required for push notification handler"
            )
            crewai_event_bus.emit(
                agent_branch,
                A2AConnectionErrorEvent(
                    endpoint=endpoint or "",
                    error=error_msg,
                    error_type="configuration_error",
                    a2a_agent_name=a2a_agent_name,
                    operation="push_notification",
                    context_id=context_id,
                    task_id=task_id,
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )
            return TaskStateResult(
                status=TaskState.failed,
                error=error_msg,
                history=new_messages,
            )

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
            )

            if not isinstance(result_or_task_id, str):
                return result_or_task_id

            task_id = result_or_task_id

            crewai_event_bus.emit(
                agent_branch,
                A2APushNotificationRegisteredEvent(
                    task_id=task_id,
                    context_id=context_id,
                    callback_url=str(config.url),
                    endpoint=endpoint,
                    a2a_agent_name=a2a_agent_name,
                    from_task=from_task,
                    from_agent=from_agent,
                ),
            )

            logger.debug(
                "Push notification callback for task %s configured at %s (via initial request)",
                task_id,
                config.url,
            )

            final_task = await _wait_for_push_result(
                task_id=task_id,
                result_store=result_store,
                timeout=polling_timeout,
                poll_interval=polling_interval,
                agent_branch=agent_branch,
                from_task=from_task,
                from_agent=from_agent,
                context_id=context_id,
                endpoint=endpoint,
                a2a_agent_name=a2a_agent_name,
            )

            if final_task is None:
                return TaskStateResult(
                    status=TaskState.failed,
                    error=f"Push notification timeout after {polling_timeout}s",
                    history=new_messages,
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
                    operation="push_notification",
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
            error_msg = f"Unexpected error during push notification: {e!s}"

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
                    operation="push_notification",
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
