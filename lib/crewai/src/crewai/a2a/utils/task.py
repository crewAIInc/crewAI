"""A2A task utilities for server-side task management."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Coroutine
from datetime import datetime
from functools import wraps
import logging
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast
from urllib.parse import urlparse

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Message,
    Task as A2ATask,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_text_artifact
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


def _parse_redis_url(url: str) -> dict[str, Any]:
    """Parse a Redis URL into aiocache configuration.

    Args:
        url: Redis connection URL (e.g., redis://localhost:6379/0).

    Returns:
        Configuration dict for aiocache.RedisCache.
    """

    parsed = urlparse(url)
    config: dict[str, Any] = {
        "cache": "aiocache.RedisCache",
        "endpoint": parsed.hostname or "localhost",
        "port": parsed.port or 6379,
    }
    if parsed.path and parsed.path != "/":
        try:
            config["db"] = int(parsed.path.lstrip("/"))
        except ValueError:
            pass
    if parsed.password:
        config["password"] = parsed.password
    return config


_redis_url = os.environ.get("REDIS_URL")

caches.set_config(
    {
        "default": _parse_redis_url(_redis_url)
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
            except (OSError, ConnectionError) as e:
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
            A2AServerTaskFailedEvent(
                task_id="",
                context_id="",
                error=msg,
                from_agent=agent,
            ),
        )
        raise ServerError(InvalidParamsError(message=msg)) from None

    task = Task(
        description=user_message,
        expected_output="Response to the user's request",
        agent=agent,
    )

    crewai_event_bus.emit(
        agent,
        A2AServerTaskStartedEvent(
            task_id=task_id,
            context_id=context_id,
            from_task=task,
            from_agent=agent,
        ),
    )

    try:
        result = await agent.aexecute_task(task=task, tools=agent.tools)
        result_str = str(result)
        history: list[Message] = [context.message] if context.message else []
        history.append(new_agent_text_message(result_str, context_id, task_id))
        await event_queue.enqueue_event(
            A2ATask(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.input_required),
                artifacts=[new_text_artifact(result_str, f"result_{task_id}")],
                history=history,
            )
        )
        crewai_event_bus.emit(
            agent,
            A2AServerTaskCompletedEvent(
                task_id=task_id,
                context_id=context_id,
                result=str(result),
                from_task=task,
                from_agent=agent,
            ),
        )
    except asyncio.CancelledError:
        crewai_event_bus.emit(
            agent,
            A2AServerTaskCanceledEvent(
                task_id=task_id,
                context_id=context_id,
                from_task=task,
                from_agent=agent,
            ),
        )
        raise
    except Exception as e:
        crewai_event_bus.emit(
            agent,
            A2AServerTaskFailedEvent(
                task_id=task_id,
                context_id=context_id,
                error=str(e),
                from_task=task,
                from_agent=agent,
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


def list_tasks(
    tasks: list[A2ATask],
    context_id: str | None = None,
    status: TaskState | None = None,
    status_timestamp_after: datetime | None = None,
    page_size: int = 50,
    page_token: str | None = None,
    history_length: int | None = None,
    include_artifacts: bool = False,
) -> tuple[list[A2ATask], str | None, int]:
    """Filter and paginate A2A tasks.

    Provides filtering by context, status, and timestamp, along with
    cursor-based pagination. This is a pure utility function that operates
    on an in-memory list of tasks - storage retrieval is handled separately.

    Args:
        tasks: All tasks to filter.
        context_id: Filter by context ID to get tasks in a conversation.
        status: Filter by task state (e.g., completed, working).
        status_timestamp_after: Filter to tasks updated after this time.
        page_size: Maximum tasks per page (default 50).
        page_token: Base64-encoded cursor from previous response.
        history_length: Limit history messages per task (None = full history).
        include_artifacts: Whether to include task artifacts (default False).

    Returns:
        Tuple of (filtered_tasks, next_page_token, total_count).
        - filtered_tasks: Tasks matching filters, paginated and trimmed.
        - next_page_token: Token for next page, or None if no more pages.
        - total_count: Total number of tasks matching filters (before pagination).
    """
    filtered: list[A2ATask] = []
    for task in tasks:
        if context_id and task.context_id != context_id:
            continue
        if status and task.status.state != status:
            continue
        if status_timestamp_after and task.status.timestamp:
            ts = datetime.fromisoformat(task.status.timestamp.replace("Z", "+00:00"))
            if ts <= status_timestamp_after:
                continue
        filtered.append(task)

    def get_timestamp(t: A2ATask) -> datetime:
        """Extract timestamp from task status for sorting."""
        if t.status.timestamp is None:
            return datetime.min
        return datetime.fromisoformat(t.status.timestamp.replace("Z", "+00:00"))

    filtered.sort(key=get_timestamp, reverse=True)
    total = len(filtered)

    start = 0
    if page_token:
        try:
            cursor_id = base64.b64decode(page_token).decode()
            for idx, task in enumerate(filtered):
                if task.id == cursor_id:
                    start = idx + 1
                    break
        except (ValueError, UnicodeDecodeError):
            pass

    page = filtered[start : start + page_size]

    result: list[A2ATask] = []
    for task in page:
        task = task.model_copy(deep=True)
        if history_length is not None and task.history:
            task.history = task.history[-history_length:]
        if not include_artifacts:
            task.artifacts = None
        result.append(task)

    next_token: str | None = None
    if result and len(result) == page_size:
        next_token = base64.b64encode(result[-1].id.encode()).decode()

    return result, next_token, total
