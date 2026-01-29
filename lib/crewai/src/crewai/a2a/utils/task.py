"""A2A task utilities for server-side task management."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Coroutine
from datetime import datetime
from functools import wraps
import json
import logging
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, TypedDict, cast
from urllib.parse import urlparse

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    FileWithBytes,
    FileWithUri,
    InternalError,
    InvalidParamsError,
    Message,
    Part,
    Task as A2ATask,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import (
    get_data_parts,
    get_file_parts,
    new_agent_text_message,
    new_data_artifact,
    new_text_artifact,
)
from a2a.utils.errors import ServerError
from aiocache import SimpleMemoryCache, caches  # type: ignore[import-untyped]
from pydantic import BaseModel

from crewai.a2a.utils.agent_card import _get_server_config
from crewai.a2a.utils.content_type import validate_message_parts
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.a2a_events import (
    A2AServerTaskCanceledEvent,
    A2AServerTaskCompletedEvent,
    A2AServerTaskFailedEvent,
    A2AServerTaskStartedEvent,
)
from crewai.task import Task
from crewai.utilities.pydantic_schema_utils import create_model_from_schema


if TYPE_CHECKING:
    from crewai.a2a.extensions.server import ExtensionContext, ServerExtensionRegistry
    from crewai.agent import Agent


logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RedisCacheConfig(TypedDict, total=False):
    """Configuration for aiocache Redis backend."""

    cache: str
    endpoint: str
    port: int
    db: int
    password: str


def _parse_redis_url(url: str) -> RedisCacheConfig:
    """Parse a Redis URL into aiocache configuration.

    Args:
        url: Redis connection URL (e.g., redis://localhost:6379/0).

    Returns:
        Configuration dict for aiocache.RedisCache.
    """
    parsed = urlparse(url)
    config: RedisCacheConfig = {
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
                logger.warning(
                    "Cancel watcher Redis error, falling back to polling",
                    extra={"task_id": task_id, "error": str(e)},
                )
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


def _convert_a2a_files_to_file_inputs(
    a2a_files: list[FileWithBytes | FileWithUri],
) -> dict[str, Any]:
    """Convert a2a file types to crewai FileInput dict.

    Args:
        a2a_files: List of FileWithBytes or FileWithUri from a2a SDK.

    Returns:
        Dictionary mapping file names to FileInput objects.
    """
    try:
        from crewai_files import File, FileBytes
    except ImportError:
        logger.debug("crewai_files not installed, returning empty file dict")
        return {}

    file_dict: dict[str, Any] = {}
    for idx, a2a_file in enumerate(a2a_files):
        if isinstance(a2a_file, FileWithBytes):
            file_bytes = base64.b64decode(a2a_file.bytes)
            name = a2a_file.name or f"file_{idx}"
            file_source = FileBytes(data=file_bytes, filename=a2a_file.name)
            file_dict[name] = File(source=file_source)
        elif isinstance(a2a_file, FileWithUri):
            name = a2a_file.name or f"file_{idx}"
            file_dict[name] = File(source=a2a_file.uri)

    return file_dict


def _extract_response_schema(parts: list[Part]) -> dict[str, Any] | None:
    """Extract response schema from message parts metadata.

    The client may include a JSON schema in TextPart metadata to specify
    the expected response format (see delegation.py line 463).

    Args:
        parts: List of message parts.

    Returns:
        JSON schema dict if found, None otherwise.
    """
    for part in parts:
        if part.root.kind == "text" and part.root.metadata:
            schema = part.root.metadata.get("schema")
            if schema and isinstance(schema, dict):
                return schema  # type: ignore[no-any-return]
    return None


def _create_result_artifact(
    result: Any,
    task_id: str,
) -> Artifact:
    """Create artifact from task result, using DataPart for structured data.

    Args:
        result: The task execution result.
        task_id: The task ID for naming the artifact.

    Returns:
        Artifact with appropriate part type (DataPart for dict/Pydantic, TextPart for strings).
    """
    artifact_name = f"result_{task_id}"
    if isinstance(result, dict):
        return new_data_artifact(artifact_name, result)
    if isinstance(result, BaseModel):
        return new_data_artifact(artifact_name, result.model_dump())
    return new_text_artifact(artifact_name, str(result))


def _build_task_description(
    user_message: str,
    structured_inputs: list[dict[str, Any]],
) -> str:
    """Build task description including structured data if present.

    Args:
        user_message: The original user message text.
        structured_inputs: List of structured data from DataParts.

    Returns:
        Task description with structured data appended if present.
    """
    if not structured_inputs:
        return user_message

    structured_json = json.dumps(structured_inputs, indent=2)
    return f"{user_message}\n\nStructured Data:\n{structured_json}"


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
    """
    await _execute_impl(agent, context, event_queue, None, None)


@cancellable
async def _execute_impl(
    agent: Agent,
    context: RequestContext,
    event_queue: EventQueue,
    extension_registry: ServerExtensionRegistry | None,
    extension_context: ExtensionContext | None,
) -> None:
    """Internal implementation for task execution with optional extensions."""
    server_config = _get_server_config(agent)
    if context.message and context.message.parts and server_config:
        allowed_modes = server_config.default_input_modes
        invalid_types = validate_message_parts(context.message.parts, allowed_modes)
        if invalid_types:
            raise ServerError(
                InvalidParamsError(
                    message=f"Unsupported content type(s): {', '.join(invalid_types)}. "
                    f"Supported: {', '.join(allowed_modes)}"
                )
            )

    if extension_registry and extension_context:
        await extension_registry.invoke_on_request(extension_context)

    user_message = context.get_user_input()

    response_model: type[BaseModel] | None = None
    structured_inputs: list[dict[str, Any]] = []
    a2a_files: list[FileWithBytes | FileWithUri] = []

    if context.message and context.message.parts:
        schema = _extract_response_schema(context.message.parts)
        if schema:
            try:
                response_model = create_model_from_schema(schema)
            except Exception as e:
                logger.debug(
                    "Failed to create response model from schema",
                    extra={"error": str(e), "schema_title": schema.get("title")},
                )

        structured_inputs = get_data_parts(context.message.parts)
        a2a_files = get_file_parts(context.message.parts)

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
        description=_build_task_description(user_message, structured_inputs),
        expected_output="Response to the user's request",
        agent=agent,
        response_model=response_model,
        input_files=_convert_a2a_files_to_file_inputs(a2a_files),
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
        if extension_registry and extension_context:
            result = await extension_registry.invoke_on_response(
                extension_context, result
            )
        result_str = str(result)
        history: list[Message] = [context.message] if context.message else []
        history.append(new_agent_text_message(result_str, context_id, task_id))
        await event_queue.enqueue_event(
            A2ATask(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.completed),
                artifacts=[_create_result_artifact(result, task_id)],
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


async def execute_with_extensions(
    agent: Agent,
    context: RequestContext,
    event_queue: EventQueue,
    extension_registry: ServerExtensionRegistry,
    extension_context: ExtensionContext,
) -> None:
    """Execute an A2A task with extension hooks.

    Args:
        agent: The CrewAI agent to execute the task.
        context: The A2A request context containing the user's message.
        event_queue: The event queue for sending responses back.
        extension_registry: Registry of server extensions.
        extension_context: Context for extension hooks.
    """
    await _execute_impl(
        agent, context, event_queue, extension_registry, extension_context
    )


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
