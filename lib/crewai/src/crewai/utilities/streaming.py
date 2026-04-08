"""Streaming utilities for crew and flow execution."""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
import contextvars
import queue
import threading
from typing import Any, NamedTuple

from typing_extensions import TypedDict

from crewai.events.base_events import BaseEvent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import LLMStreamChunkEvent
from crewai.types.streaming import (
    CrewStreamingOutput,
    FlowStreamingOutput,
    StreamChunk,
    StreamChunkType,
    ToolCallChunk,
)
from crewai.utilities.string_utils import sanitize_tool_name


class TaskInfo(TypedDict):
    """Task context information for streaming."""

    index: int
    name: str
    id: str
    agent_role: str
    agent_id: str


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
                tool_name=sanitize_tool_name(event.tool_call.function.name)
                if event.tool_call.function.name
                else None,
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
        handlers: frozenset[Callable[..., None]] = crewai_event_bus._sync_handlers.get(
            LLMStreamChunkEvent, frozenset()
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
) -> Iterator[StreamChunk]:
    """Create a chunk generator that uses a holder to access streaming output.

    Args:
        state: The streaming state.
        run_func: Function to run in a separate thread.
        output_holder: Single-element list that will contain the streaming output.

    Yields:
        StreamChunk objects as they arrive.
    """
    cancel_event = threading.Event()
    ctx = contextvars.copy_context()
    thread = threading.Thread(target=ctx.run, args=(run_func,), daemon=True)
    thread.start()

    # Wire cancellation to the streaming output once the holder is populated
    def _wire_cancel() -> None:
        if output_holder:
            output_holder[0]._cancel_thread_event = cancel_event
            output_holder[0]._background_thread = thread

    try:
        while True:
            # Poll the queue with a timeout so we can check cancellation
            while True:
                _wire_cancel()
                if cancel_event.is_set():
                    return
                try:
                    item = state.sync_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not cancel_event.is_set():
            thread.join()
        if output_holder:
            _finalize_streaming(state, output_holder[0])
        else:
            _unregister_handler(state.handler)


async def create_async_chunk_generator(
    state: StreamingState,
    run_coro: Callable[[], Any],
    output_holder: list[CrewStreamingOutput | FlowStreamingOutput],
) -> AsyncIterator[StreamChunk]:
    """Create an async chunk generator that uses a holder to access streaming output.

    Args:
        state: The streaming state.
        run_coro: Coroutine function to run as a task.
        output_holder: Single-element list that will contain the streaming output.

    Yields:
        StreamChunk objects as they arrive.
    """
    if state.async_queue is None:
        raise RuntimeError(
            "Async queue not initialized. Use create_streaming_state(use_async=True)."
        )

    cancel_event = asyncio.Event()
    task = asyncio.create_task(run_coro())

    # Wire cancellation to the streaming output once the holder is populated
    def _wire_cancel() -> None:
        if output_holder:
            output_holder[0]._cancel_event = cancel_event
            output_holder[0]._background_task = task

    try:
        while True:
            _wire_cancel()
            # Use asyncio.wait to race between the queue and cancellation
            get_task = asyncio.ensure_future(state.async_queue.get())
            cancel_wait = asyncio.ensure_future(cancel_event.wait())
            done, pending = await asyncio.wait(
                {get_task, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
                try:
                    await p
                except (asyncio.CancelledError, Exception):  # noqa: S110
                    pass
            if cancel_wait in done:
                # Cancellation was requested
                return
            item = get_task.result()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not cancel_event.is_set():
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: S110
                pass
        else:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: S110
                    pass
        if output_holder:
            _finalize_streaming(state, output_holder[0])
        else:
            _unregister_handler(state.handler)
