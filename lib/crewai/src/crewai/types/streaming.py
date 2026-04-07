"""Streaming output types for crew and flow execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from enum import Enum
import threading
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from crewai.crews.crew_output import CrewOutput


T = TypeVar("T")


class StreamChunkType(Enum):
    """Type of streaming chunk."""

    TEXT = "text"
    TOOL_CALL = "tool_call"


class ToolCallChunk(BaseModel):
    """Tool call information in a streaming chunk.

    Attributes:
        tool_id: Unique identifier for the tool call
        tool_name: Name of the tool being called
        arguments: JSON string of tool arguments
        index: Index of the tool call in the response
    """

    tool_id: str | None = None
    tool_name: str | None = None
    arguments: str = ""
    index: int = 0


class StreamChunk(BaseModel):
    """Base streaming chunk with full context.

    Attributes:
        content: The streaming content (text or partial content)
        chunk_type: Type of the chunk (text, tool_call, etc.)
        task_index: Index of the current task (0-based)
        task_name: Name or description of the current task
        task_id: Unique identifier of the task
        agent_role: Role of the agent executing the task
        agent_id: Unique identifier of the agent
        tool_call: Tool call information if chunk_type is TOOL_CALL
    """

    content: str = Field(description="The streaming content")
    chunk_type: StreamChunkType = Field(
        default=StreamChunkType.TEXT, description="Type of the chunk"
    )
    task_index: int = Field(default=0, description="Index of the current task")
    task_name: str = Field(default="", description="Name of the current task")
    task_id: str = Field(default="", description="Unique identifier of the task")
    agent_role: str = Field(default="", description="Role of the agent")
    agent_id: str = Field(default="", description="Unique identifier of the agent")
    tool_call: ToolCallChunk | None = Field(
        default=None, description="Tool call information"
    )

    def __str__(self) -> str:
        """Return the chunk content as a string."""
        return self.content


class StreamingOutputBase(Generic[T]):
    """Base class for streaming output with result access.

    Provides iteration over stream chunks and access to final result
    via the .result property after streaming completes.

    Supports graceful cancellation via ``aclose()`` (async) and ``cancel()``
    (sync).  When cancelled, in-flight background tasks are aborted and
    resources are released promptly.
    """

    def __init__(self) -> None:
        """Initialize streaming output base."""
        self._result: T | None = None
        self._completed: bool = False
        self._cancelled: bool = False
        self._chunks: list[StreamChunk] = []
        self._error: Exception | None = None
        self._cancel_event: asyncio.Event | None = None
        self._cancel_thread_event: threading.Event | None = None
        self._background_task: asyncio.Task[Any] | None = None
        self._background_thread: threading.Thread | None = None

    @property
    def result(self) -> T:
        """Get the final result after streaming completes.

        Returns:
            The final output (CrewOutput for crews, Any for flows).

        Raises:
            RuntimeError: If streaming has not completed yet.
            Exception: If streaming failed with an error.
        """
        if not self._completed:
            raise RuntimeError(
                "Streaming has not completed yet. "
                "Iterate over all chunks before accessing result."
            )
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("No result available")
        return self._result

    @property
    def is_completed(self) -> bool:
        """Check if streaming has completed."""
        return self._completed

    @property
    def is_cancelled(self) -> bool:
        """Check if streaming was cancelled."""
        return self._cancelled

    @property
    def chunks(self) -> list[StreamChunk]:
        """Get all collected chunks so far."""
        return self._chunks.copy()

    def get_full_text(self) -> str:
        """Get all streamed text content concatenated.

        Returns:
            All text chunks concatenated together.
        """
        return "".join(
            chunk.content
            for chunk in self._chunks
            if chunk.chunk_type == StreamChunkType.TEXT
        )

    async def aclose(self) -> None:
        """Cancel streaming and clean up resources.

        Signals cancellation to the background task, waits briefly for it
        to finish, and marks the stream as completed and cancelled.
        Safe to call multiple times or on an already-completed stream.

        Example:
            ```python
            streaming = await crew.akickoff(inputs=inputs)
            try:
                async for chunk in streaming:
                    ...
            finally:
                await streaming.aclose()
            ```
        """
        if self._completed:
            return

        self._cancelled = True

        if self._cancel_event is not None:
            self._cancel_event.set()

        if self._cancel_thread_event is not None:
            self._cancel_thread_event.set()

        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except (asyncio.CancelledError, Exception):  # noqa: S110
                pass

        self._completed = True

    def cancel(self) -> None:
        """Synchronously cancel streaming and clean up resources.

        Signals cancellation to the background thread/task and marks the
        stream as completed and cancelled.  For async contexts prefer
        ``aclose()`` which can ``await`` background cleanup.

        Example:
            ```python
            streaming = crew.kickoff(inputs=inputs)
            try:
                for chunk in streaming:
                    ...
            finally:
                streaming.cancel()
            ```
        """
        if self._completed:
            return

        self._cancelled = True

        if self._cancel_event is not None:
            self._cancel_event.set()

        if self._cancel_thread_event is not None:
            self._cancel_thread_event.set()

        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()

        self._completed = True


class CrewStreamingOutput(StreamingOutputBase["CrewOutput"]):
    """Streaming output wrapper for crew execution.

    Provides both sync and async iteration over stream chunks,
    with access to the final CrewOutput via the .result property.

    For kickoff_for_each_async with streaming, use .results to get list of outputs.

    Example:
        ```python
        # Single crew
        streaming = crew.kickoff(inputs={"topic": "AI"})
        for chunk in streaming:
            print(chunk.content, end="", flush=True)
        result = streaming.result

        # Multiple crews (kickoff_for_each_async)
        streaming = await crew.kickoff_for_each_async(
            [{"topic": "AI"}, {"topic": "ML"}]
        )
        async for chunk in streaming:
            print(chunk.content, end="", flush=True)
        results = streaming.results  # List of CrewOutput
        ```
    """

    def __init__(
        self,
        sync_iterator: Iterator[StreamChunk] | None = None,
        async_iterator: AsyncIterator[StreamChunk] | None = None,
    ) -> None:
        """Initialize crew streaming output.

        Args:
            sync_iterator: Synchronous iterator for chunks.
            async_iterator: Asynchronous iterator for chunks.
        """
        super().__init__()
        self._sync_iterator = sync_iterator
        self._async_iterator = async_iterator
        self._results: list[CrewOutput] | None = None

    @property
    def results(self) -> list[CrewOutput]:
        """Get all results for kickoff_for_each_async.

        Returns:
            List of CrewOutput from all crews.

        Raises:
            RuntimeError: If streaming has not completed or results not available.
        """
        if not self._completed:
            raise RuntimeError(
                "Streaming has not completed yet. "
                "Iterate over all chunks before accessing results."
            )
        if self._error is not None:
            raise self._error
        if self._results is not None:
            return self._results
        if self._result is not None:
            return [self._result]
        raise RuntimeError("No results available")

    def _set_results(self, results: list[CrewOutput]) -> None:
        """Set multiple results for kickoff_for_each_async.

        Args:
            results: List of CrewOutput from all crews.
        """
        self._results = results
        self._completed = True

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over stream chunks synchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If sync iterator not available.
        """
        if self._sync_iterator is None:
            raise RuntimeError("Sync iterator not available")
        try:
            for chunk in self._sync_iterator:
                self._chunks.append(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Return async iterator for stream chunks.

        Returns:
            Async iterator for StreamChunk objects.
        """
        return self._async_iterate()

    async def _async_iterate(self) -> AsyncIterator[StreamChunk]:
        """Iterate over stream chunks asynchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If async iterator not available.
        """
        if self._async_iterator is None:
            raise RuntimeError("Async iterator not available")
        try:
            async for chunk in self._async_iterator:
                self._chunks.append(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def _set_result(self, result: CrewOutput) -> None:
        """Set the final result after streaming completes.

        Args:
            result: The final CrewOutput.
        """
        self._result = result
        self._completed = True


class FlowStreamingOutput(StreamingOutputBase[Any]):
    """Streaming output wrapper for flow execution.

    Provides both sync and async iteration over stream chunks,
    with access to the final flow output via the .result property.

    Example:
        ```python
        # Sync usage
        streaming = flow.kickoff_streaming()
        for chunk in streaming:
            print(chunk.content, end="", flush=True)
        result = streaming.result

        # Async usage
        streaming = await flow.kickoff_streaming_async()
        async for chunk in streaming:
            print(chunk.content, end="", flush=True)
        result = streaming.result
        ```
    """

    def __init__(
        self,
        sync_iterator: Iterator[StreamChunk] | None = None,
        async_iterator: AsyncIterator[StreamChunk] | None = None,
    ) -> None:
        """Initialize flow streaming output.

        Args:
            sync_iterator: Synchronous iterator for chunks.
            async_iterator: Asynchronous iterator for chunks.
        """
        super().__init__()
        self._sync_iterator = sync_iterator
        self._async_iterator = async_iterator

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over stream chunks synchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If sync iterator not available.
        """
        if self._sync_iterator is None:
            raise RuntimeError("Sync iterator not available")
        try:
            for chunk in self._sync_iterator:
                self._chunks.append(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Return async iterator for stream chunks.

        Returns:
            Async iterator for StreamChunk objects.
        """
        return self._async_iterate()

    async def _async_iterate(self) -> AsyncIterator[StreamChunk]:
        """Iterate over stream chunks asynchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If async iterator not available.
        """
        if self._async_iterator is None:
            raise RuntimeError("Async iterator not available")
        try:
            async for chunk in self._async_iterator:
                self._chunks.append(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def _set_result(self, result: Any) -> None:
        """Set the final result after streaming completes.

        Args:
            result: The final flow output.
        """
        self._result = result
        self._completed = True
