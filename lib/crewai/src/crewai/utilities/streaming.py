"""Streaming utilities for crew and flow execution."""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
import queue
import threading
import time
from typing import Any, NamedTuple

from typing_extensions import TypedDict

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMStreamChunkEvent
from crewai.types.streaming import (
    AgentInfoChunk,
    CrewStreamingOutput,
    FlowStreamingOutput,
    StreamChunk,
    StreamChunkType,
    TaskInfoChunk,
    ToolCallChunk,
)


@dataclass
class StreamingConfig:
    """Configuration for streaming behavior with slow LLM models.

    Attributes:
        heartbeat_interval: Seconds between heartbeat emissions (default: 30)
        queue_poll_timeout: Seconds to wait for queue items before checking for heartbeat (default: 1.0)
        llm_timeout: Maximum seconds to wait for LLM response (None = wait forever)
        max_retries: Number of retries for transient failures (default: 3)
        retry_delay: Seconds to wait between retries (default: 1.0)
        retry_backoff: Multiplier for retry delay (default: 2.0)
    """

    heartbeat_interval: float = 30.0
    queue_poll_timeout: float = 1.0
    llm_timeout: float | None = None  # None = wait forever (for slow local models)
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0


# Default global config
DEFAULT_STREAMING_CONFIG = StreamingConfig()


class HeartbeatTracker:
    """Tracks time since last heartbeat to know when to emit.

    Used to send periodic "still waiting" signals to the UI during
    long LLM response times (common with local models).
    """

    def __init__(self, interval: float = 30.0) -> None:
        """Initialize the heartbeat tracker.

        Args:
            interval: Seconds between heartbeat emissions.
        """
        self.interval = interval
        self.last_heartbeat = time.time()
        self.wait_start = time.time()

    def should_emit(self) -> bool:
        """Check if enough time has passed to emit a heartbeat."""
        return time.time() - self.last_heartbeat >= self.interval

    def mark_emitted(self) -> None:
        """Mark that a heartbeat was just emitted."""
        self.last_heartbeat = time.time()

    def get_wait_duration(self) -> float:
        """Get total time spent waiting."""
        return time.time() - self.wait_start

    def reset(self) -> None:
        """Reset tracker for new wait period."""
        self.last_heartbeat = time.time()
        self.wait_start = time.time()


def create_heartbeat_chunk(
    current_task_info: "TaskInfo",
    wait_duration_seconds: float,
) -> StreamChunk:
    """Create a HEARTBEAT chunk to indicate system is still processing.

    Args:
        current_task_info: Current task context info.
        wait_duration_seconds: How long we've been waiting.

    Returns:
        StreamChunk with HEARTBEAT type.
    """
    return StreamChunk(
        content=f"Still waiting for LLM response... ({wait_duration_seconds:.0f}s)",
        chunk_type=StreamChunkType.HEARTBEAT,
        task_index=current_task_info["index"],
        task_name=current_task_info["name"],
        task_id=current_task_info["id"],
        agent_role=current_task_info["agent_role"],
        agent_id=current_task_info["agent_id"],
        timestamp=datetime.now(),
    )


class TaskInfo(TypedDict):
    """Task context information for streaming."""

    index: int
    name: str
    id: str
    agent_role: str
    agent_id: str


def update_task_info(
    current_task_info: TaskInfo,
    task_index: int,
    task_name: str,
    task_id: str,
    agent_role: str = "",
    agent_id: str = "",
) -> None:
    """Update current task info for streaming context.

    This function updates the mutable TaskInfo dictionary so that
    streaming chunks contain correct task/agent information.

    Args:
        current_task_info: The TaskInfo dict to update (mutable)
        task_index: Index of the current task (0-based)
        task_name: Name or description of the task
        task_id: Unique identifier of the task
        agent_role: Role of the agent executing the task
        agent_id: Unique identifier of the agent
    """
    current_task_info["index"] = task_index
    current_task_info["name"] = task_name
    current_task_info["id"] = task_id
    current_task_info["agent_role"] = agent_role
    current_task_info["agent_id"] = agent_id


def create_task_started_chunk(
    task: Any,
    task_index: int,
    total_tasks: int,
    agent: Any,
) -> StreamChunk:
    """Create a TASK_STARTED chunk for UI lifecycle tracking.

    Args:
        task: The task being started.
        task_index: Index of the task (0-based).
        total_tasks: Total number of tasks in the crew.
        agent: The agent executing the task.

    Returns:
        StreamChunk with TASK_STARTED type and task/agent info.
    """
    task_name = task.name or task.description[:50] if task.description else ""
    return StreamChunk(
        content=f"Starting task: {task_name}",
        chunk_type=StreamChunkType.TASK_STARTED,
        task_index=task_index,
        task_name=task_name,
        task_id=str(task.id),
        agent_role=agent.role if agent else "",
        agent_id=str(agent.id) if agent else "",
        task_info=TaskInfoChunk(
            task_index=task_index,
            task_name=task_name,
            task_id=str(task.id),
            expected_output=task.expected_output or "",
            total_tasks=total_tasks,
        ),
        agent_info=AgentInfoChunk(
            agent_role=agent.role if agent else "",
            agent_id=str(agent.id) if agent else "",
            agent_goal=agent.goal if agent else "",
        ),
    )


def create_task_completed_chunk(
    task: Any,
    task_index: int,
    task_output: Any,
) -> StreamChunk:
    """Create a TASK_COMPLETED chunk for UI lifecycle tracking.

    Args:
        task: The task that completed.
        task_index: Index of the task (0-based).
        task_output: The output from the task.

    Returns:
        StreamChunk with TASK_COMPLETED type and output info.
    """
    task_name = task.name or task.description[:50] if task.description else ""
    output_str = str(task_output.raw) if hasattr(task_output, "raw") else str(task_output)
    return StreamChunk(
        content=f"Completed task: {task_name}",
        chunk_type=StreamChunkType.TASK_COMPLETED,
        task_index=task_index,
        task_name=task_name,
        task_id=str(task.id),
        agent_role=task_output.agent if hasattr(task_output, "agent") else "",
        agent_id="",
        task_info=TaskInfoChunk(
            task_index=task_index,
            task_name=task_name,
            task_id=str(task.id),
            output=output_str[:500] if len(output_str) > 500 else output_str,
        ),
    )


def emit_lifecycle_chunk(
    state: "StreamingState",
    chunk: StreamChunk,
    is_async: bool = False,
) -> None:
    """Emit a lifecycle chunk to the stream queue.

    Args:
        state: The streaming state.
        chunk: The lifecycle chunk to emit.
        is_async: Whether this is an async stream.
    """
    if is_async and state.async_queue is not None and state.loop is not None:
        state.loop.call_soon_threadsafe(state.async_queue.put_nowait, chunk)
    else:
        state.sync_queue.put(chunk)


class StreamingState(NamedTuple):
    """Immutable state for streaming execution."""

    current_task_info: TaskInfo
    result_holder: list[Any]
    sync_queue: queue.Queue[StreamChunk | None | Exception]
    async_queue: asyncio.Queue[StreamChunk | None | Exception] | None
    loop: asyncio.AbstractEventLoop | None
    handler: Callable[[Any, BaseEvent], None]


def _extract_tool_call_info(
    event: LLMStreamChunkEvent,
) -> tuple[StreamChunkType, ToolCallChunk | None]:
    """Extract tool call information from an LLM stream chunk event.

    Args:
        event: The LLM stream chunk event to process.

    Returns:
        A tuple of (chunk_type, tool_call_chunk) where tool_call_chunk is None
        if the event is not a tool call.
    """
    if event.tool_call:
        return (
            StreamChunkType.TOOL_CALL,
            ToolCallChunk(
                tool_id=event.tool_call.id,
                tool_name=event.tool_call.function.name,
                arguments=event.tool_call.function.arguments,
                index=event.tool_call.index,
            ),
        )
    return StreamChunkType.TEXT, None


def _create_stream_chunk(
    event: LLMStreamChunkEvent,
    current_task_info: TaskInfo,
) -> StreamChunk:
    """Create a StreamChunk from an LLM stream chunk event.

    Args:
        event: The LLM stream chunk event to process.
        current_task_info: Task context info.

    Returns:
        A StreamChunk populated with event and task info.
    """
    chunk_type, tool_call_chunk = _extract_tool_call_info(event)

    return StreamChunk(
        content=event.chunk,
        chunk_type=chunk_type,
        task_index=current_task_info["index"],
        task_name=current_task_info["name"],
        task_id=current_task_info["id"],
        agent_role=event.agent_role or current_task_info["agent_role"],
        agent_id=event.agent_id or current_task_info["agent_id"],
        tool_call=tool_call_chunk,
    )


def _create_stream_handler(
    current_task_info: TaskInfo,
    sync_queue: queue.Queue[StreamChunk | None | Exception],
    async_queue: asyncio.Queue[StreamChunk | None | Exception] | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> Callable[[Any, BaseEvent], None]:
    """Create a stream handler function.

    Args:
        current_task_info: Task context info.
        sync_queue: Synchronous queue for chunks.
        async_queue: Optional async queue for chunks.
        loop: Optional event loop for async operations.

    Returns:
        Handler function that can be registered with the event bus.
    """

    def stream_handler(_: Any, event: BaseEvent) -> None:
        """Handle LLM stream chunk events and enqueue them.

        Args:
            _: Event source (unused).
            event: The event to process.
        """
        if not isinstance(event, LLMStreamChunkEvent):
            return

        chunk = _create_stream_chunk(event, current_task_info)

        if async_queue is not None and loop is not None:
            loop.call_soon_threadsafe(async_queue.put_nowait, chunk)
        else:
            sync_queue.put(chunk)

    return stream_handler


def _unregister_handler(handler: Callable[[Any, BaseEvent], None]) -> None:
    """Unregister a stream handler from the event bus.

    Args:
        handler: The handler function to unregister.
    """
    with crewai_event_bus._rwlock.w_locked():
        handlers: frozenset[Callable[[Any, BaseEvent], None]] = (
            crewai_event_bus._sync_handlers.get(LLMStreamChunkEvent, frozenset())
        )
        crewai_event_bus._sync_handlers[LLMStreamChunkEvent] = handlers - {handler}


def _finalize_streaming(
    state: StreamingState,
    streaming_output: CrewStreamingOutput | FlowStreamingOutput,
) -> None:
    """Finalize streaming by unregistering handler and setting result.

    Args:
        state: The streaming state to finalize.
        streaming_output: The streaming output to set the result on.
    """
    _unregister_handler(state.handler)
    if state.result_holder:
        streaming_output._set_result(state.result_holder[0])


def create_streaming_state(
    current_task_info: TaskInfo,
    result_holder: list[Any],
    use_async: bool = False,
) -> StreamingState:
    """Create and register streaming state.

    Args:
        current_task_info: Task context info.
        result_holder: List to hold the final result.
        use_async: Whether to use async queue.

    Returns:
        Initialized StreamingState with registered handler.
    """
    sync_queue: queue.Queue[StreamChunk | None | Exception] = queue.Queue()
    async_queue: asyncio.Queue[StreamChunk | None | Exception] | None = None
    loop: asyncio.AbstractEventLoop | None = None

    if use_async:
        async_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

    handler = _create_stream_handler(current_task_info, sync_queue, async_queue, loop)
    crewai_event_bus.register_handler(LLMStreamChunkEvent, handler)

    return StreamingState(
        current_task_info=current_task_info,
        result_holder=result_holder,
        sync_queue=sync_queue,
        async_queue=async_queue,
        loop=loop,
        handler=handler,
    )


def signal_end(state: StreamingState, is_async: bool = False) -> None:
    """Signal end of stream.

    Args:
        state: The streaming state.
        is_async: Whether this is an async stream.
    """
    if is_async and state.async_queue is not None and state.loop is not None:
        state.loop.call_soon_threadsafe(state.async_queue.put_nowait, None)
    else:
        state.sync_queue.put(None)


def signal_error(
    state: StreamingState, error: Exception, is_async: bool = False
) -> None:
    """Signal an error in the stream.

    Args:
        state: The streaming state.
        error: The exception to signal.
        is_async: Whether this is an async stream.
    """
    if is_async and state.async_queue is not None and state.loop is not None:
        state.loop.call_soon_threadsafe(state.async_queue.put_nowait, error)
    else:
        state.sync_queue.put(error)


def create_chunk_generator(
    state: StreamingState,
    run_func: Callable[[], None],
    output_holder: list[CrewStreamingOutput | FlowStreamingOutput],
    config: StreamingConfig | None = None,
) -> Iterator[StreamChunk]:
    """Create a chunk generator that uses a holder to access streaming output.

    Features timeout-based polling with heartbeat emission for slow LLM models.

    Args:
        state: The streaming state.
        run_func: Function to run in a separate thread.
        output_holder: Single-element list that will contain the streaming output.
        config: Optional streaming configuration (uses defaults if None).

    Yields:
        StreamChunk objects as they arrive, including HEARTBEAT chunks during long waits.
    """
    cfg = config or DEFAULT_STREAMING_CONFIG
    thread = threading.Thread(target=run_func, daemon=True)
    thread.start()

    heartbeat_tracker = HeartbeatTracker(interval=cfg.heartbeat_interval)

    try:
        while True:
            try:
                # Use timeout-based polling instead of blocking forever
                item = state.sync_queue.get(timeout=cfg.queue_poll_timeout)

                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                # Reset heartbeat tracker when we get actual content
                heartbeat_tracker.reset()
                yield item

            except queue.Empty:
                # No item received within timeout - check if we should emit heartbeat
                if heartbeat_tracker.should_emit():
                    heartbeat_chunk = create_heartbeat_chunk(
                        state.current_task_info,
                        heartbeat_tracker.get_wait_duration(),
                    )
                    heartbeat_tracker.mark_emitted()
                    yield heartbeat_chunk
                # Continue polling
                continue

    finally:
        # Wait for thread with a reasonable timeout to avoid blocking forever
        thread.join(timeout=5.0)
        if output_holder:
            _finalize_streaming(state, output_holder[0])
        else:
            _unregister_handler(state.handler)


async def create_async_chunk_generator(
    state: StreamingState,
    run_coro: Callable[[], Any],
    output_holder: list[CrewStreamingOutput | FlowStreamingOutput],
    config: StreamingConfig | None = None,
) -> AsyncIterator[StreamChunk]:
    """Create an async chunk generator that uses a holder to access streaming output.

    Features timeout-based polling with heartbeat emission for slow LLM models.

    Args:
        state: The streaming state.
        run_coro: Coroutine function to run as a task.
        output_holder: Single-element list that will contain the streaming output.
        config: Optional streaming configuration (uses defaults if None).

    Yields:
        StreamChunk objects as they arrive, including HEARTBEAT chunks during long waits.
    """
    if state.async_queue is None:
        raise RuntimeError(
            "Async queue not initialized. Use create_streaming_state(use_async=True)."
        )

    cfg = config or DEFAULT_STREAMING_CONFIG
    task = asyncio.create_task(run_coro())

    heartbeat_tracker = HeartbeatTracker(interval=cfg.heartbeat_interval)

    try:
        while True:
            try:
                # Use timeout-based polling instead of blocking forever
                item = await asyncio.wait_for(
                    state.async_queue.get(),
                    timeout=cfg.queue_poll_timeout,
                )

                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                # Reset heartbeat tracker when we get actual content
                heartbeat_tracker.reset()
                yield item

            except asyncio.TimeoutError:
                # No item received within timeout - check if we should emit heartbeat
                if heartbeat_tracker.should_emit():
                    heartbeat_chunk = create_heartbeat_chunk(
                        state.current_task_info,
                        heartbeat_tracker.get_wait_duration(),
                    )
                    heartbeat_tracker.mark_emitted()
                    yield heartbeat_chunk
                # Continue polling
                continue

    finally:
        await task
        if output_holder:
            _finalize_streaming(state, output_holder[0])
        else:
            _unregister_handler(state.handler)
