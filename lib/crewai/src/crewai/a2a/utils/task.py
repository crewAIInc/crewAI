"""A2A task utilities for server-side task management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from functools import wraps
import logging
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Task as A2ATask,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import completed_task, new_text_artifact
from a2a.utils.errors import ServerError
from aiocache import SimpleMemoryCache, caches  # type: ignore[import-untyped]

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AServerTaskCanceledEvent,
    A2AServerTaskCompletedEvent,
    A2AServerTaskFailedEvent,
    A2AServerTaskStartedEvent,
)
from crewai.task import Task


if TYPE_CHECKING:
    from crewai.agent import Agent


logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


_redis_url = os.environ.get("REDIS_URL")

caches.set_config(
    {
        "default": {
            "cache": "aiocache.RedisCache",
            "endpoint": _redis_url,
        }
        if _redis_url
        else {
            "cache": "aiocache.SimpleMemoryCache",
        }
    }
)


def cancellable(
    fn: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Decorator that enables cancellation for A2A task execution.

    Runs a cancellation watcher concurrently with the wrapped function.
    When a cancel event is published, the execution is cancelled.

    Args:
        fn: The async function to wrap.

    Returns:
        Wrapped function with cancellation support.
    """

    @wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        """Wrap function with cancellation monitoring."""
        context: RequestContext | None = None
        for arg in args:
            if isinstance(arg, RequestContext):
                context = arg
                break
        if context is None:
            context = cast(RequestContext | None, kwargs.get("context"))

        if context is None:
            return await fn(*args, **kwargs)

        task_id = context.task_id
        cache = caches.get("default")

        async def poll_for_cancel() -> bool:
            """Poll cache for cancellation flag."""
            while True:
                if await cache.get(f"cancel:{task_id}"):
                    return True
                await asyncio.sleep(0.1)

        async def watch_for_cancel() -> bool:
            """Watch for cancellation events via pub/sub or polling."""
            if isinstance(cache, SimpleMemoryCache):
                return await poll_for_cancel()

            try:
                client = cache.client
                pubsub = client.pubsub()
                await pubsub.subscribe(f"cancel:{task_id}")
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        return True
            except Exception as e:
                logger.warning("Cancel watcher error for task_id=%s: %s", task_id, e)
                return await poll_for_cancel()
            return False

        execute_task = asyncio.create_task(fn(*args, **kwargs))
        cancel_watch = asyncio.create_task(watch_for_cancel())

        try:
            done, _ = await asyncio.wait(
                [execute_task, cancel_watch],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_watch in done:
                execute_task.cancel()
                try:
                    await execute_task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError(f"Task {task_id} was cancelled")
            cancel_watch.cancel()
            return execute_task.result()
        finally:
            await cache.delete(f"cancel:{task_id}")

    return wrapper


@cancellable
async def execute(
    agent: Agent,
    context: RequestContext,
    event_queue: EventQueue,
) -> None:
    """Execute an A2A task using a CrewAI agent.

    Args:
        agent: The CrewAI agent to execute the task.
        context: The A2A request context containing the user's message.
        event_queue: The event queue for sending responses back.

    TODOs:
        * need to impl both of structured output and file inputs, depends on `file_inputs` for
          `crewai.task.Task`, pass the below two to Task. both utils in `a2a.utils.parts`
        * structured outputs ingestion, `structured_inputs = get_data_parts(parts=context.message.parts)`
        * file inputs ingestion, `file_inputs = get_file_parts(parts=context.message.parts)`
    """

    user_message = context.get_user_input()
    task_id = context.task_id
    context_id = context.context_id
    if task_id is None or context_id is None:
        msg = "task_id and context_id are required"
        crewai_event_bus.emit(
            agent,
            A2AServerTaskFailedEvent(a2a_task_id="", a2a_context_id="", error=msg),
        )
        raise ServerError(InvalidParamsError(message=msg)) from None

    task = Task(
        description=user_message,
        expected_output="Response to the user's request",
        agent=agent,
    )

    crewai_event_bus.emit(
        agent,
        A2AServerTaskStartedEvent(a2a_task_id=task_id, a2a_context_id=context_id),
    )

    try:
        result = await agent.aexecute_task(task=task, tools=agent.tools)
        await event_queue.enqueue_event(
            completed_task(
                task_id,
                context_id,
                [new_text_artifact(str(result), f"result_{task_id}")],
                [context.message] if context.message else [],
            )
        )
        crewai_event_bus.emit(
            agent,
            A2AServerTaskCompletedEvent(
                a2a_task_id=task_id, a2a_context_id=context_id, result=str(result)
            ),
        )
    except asyncio.CancelledError:
        crewai_event_bus.emit(
            agent,
            A2AServerTaskCanceledEvent(a2a_task_id=task_id, a2a_context_id=context_id),
        )
        raise
    except Exception as e:
        crewai_event_bus.emit(
            agent,
            A2AServerTaskFailedEvent(
                a2a_task_id=task_id, a2a_context_id=context_id, error=str(e)
            ),
        )
        raise ServerError(
            error=InternalError(message=f"Task execution failed: {e}")
        ) from e


async def cancel(
    context: RequestContext,
    event_queue: EventQueue,
) -> A2ATask | None:
    """Cancel an A2A task.

    Publishes a cancel event that the cancellable decorator listens for.

    Args:
        context: The A2A request context containing task information.
        event_queue: The event queue for sending the cancellation status.

    Returns:
        The canceled task with updated status.
    """
    task_id = context.task_id
    context_id = context.context_id
    if task_id is None or context_id is None:
        raise ServerError(InvalidParamsError(message="task_id and context_id required"))

    if context.current_task and context.current_task.status.state in (
        TaskState.completed,
        TaskState.failed,
        TaskState.canceled,
    ):
        return context.current_task

    cache = caches.get("default")

    await cache.set(f"cancel:{task_id}", True, ttl=3600)
    if not isinstance(cache, SimpleMemoryCache):
        await cache.client.publish(f"cancel:{task_id}", "cancel")

    await event_queue.enqueue_event(
        TaskStatusUpdateEvent(
            task_id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.canceled),
            final=True,
        )
    )

    if context.current_task:
        context.current_task.status = TaskStatus(state=TaskState.canceled)
        return context.current_task
    return None
