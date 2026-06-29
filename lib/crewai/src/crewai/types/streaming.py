"""Streaming output types for crew and flow execution."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import Self


if TYPE_CHECKING:
    from crewai.crews.crew_output import CrewOutput


T = TypeVar("T")
_MISSING = object()

StreamChannel = Literal[
    "llm",
    "flow",
    "tools",
    "messages",
    "lifecycle",
    "custom",
]


class StreamFrame(BaseModel):
    """Stable public stream frame emitted by streamable runtimes."""

    version: Literal["v1"] = "v1"
    id: str = Field(description="Unique frame/event identifier")
    seq: int | None = Field(default=None, description="Execution-local order")
    type: str = Field(description="Source event type")
    channel: StreamChannel = Field(description="High-level stream channel")
    namespace: list[str] = Field(default_factory=list)
    timestamp: datetime
    parent_id: str | None = None
    previous_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class StreamSessionBase(Generic[T]):
    """Base stream session with ordered frame iteration and result access."""

    def __init__(
        self,
        sync_iterator: Iterator[StreamFrame] | None = None,
        async_iterator: AsyncIterator[StreamFrame] | None = None,
    ) -> None:
        self._result: T | object = _MISSING
        self._completed = False
        self._frames: list[StreamFrame] = []
        self._error: Exception | None = None
        self._cancelled = False
        self._exhausted = False
        self._on_cleanup: Callable[[], None] | None = None
        self._sync_iterator = sync_iterator
        self._async_iterator = async_iterator

    @property
    def result(self) -> T:
        """Return the final result after stream exhaustion or completion."""
        if not self._completed:
            raise RuntimeError(
                "Streaming has not completed yet. "
                "Iterate over all frames before accessing result."
            )
        if self._error is not None:
            raise self._error
        if self._result is _MISSING:
            raise RuntimeError("No result available")
        return self._result  # type: ignore[return-value]

    @property
    def is_completed(self) -> bool:
        """Check if the stream has completed."""
        return self._completed

    @property
    def is_cancelled(self) -> bool:
        """Check if the stream was cancelled."""
        return self._cancelled

    @property
    def is_exhausted(self) -> bool:
        """Check if the stream iterator was fully consumed."""
        return self._exhausted

    @property
    def frames(self) -> list[StreamFrame]:
        """Return collected frames."""
        return self._frames.copy()

    def _set_result(self, result: T) -> None:
        self._result = result
        self._completed = True


class StreamSession(StreamSessionBase[T]):
    """Synchronous stream session for ordered public frames."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if not self._exhausted:
            self.close()

    @property
    def events(self) -> Iterator[StreamFrame]:
        """Iterate over all ordered frames."""
        return self.subscribe()

    @property
    def llm(self) -> Iterator[StreamFrame]:
        """Iterate over LLM token and thinking frames."""
        return self.subscribe(channels=["llm"])

    @property
    def messages(self) -> Iterator[StreamFrame]:
        """Iterate over conversation message frames."""
        return self.subscribe(channels=["messages"])

    @property
    def flow(self) -> Iterator[StreamFrame]:
        """Iterate over Flow lifecycle and method frames."""
        return self.subscribe(channels=["flow"])

    @property
    def tools(self) -> Iterator[StreamFrame]:
        """Iterate over tool execution frames."""
        return self.subscribe(channels=["tools"])

    def interleave(self, channels: Sequence[StreamChannel]) -> Iterator[StreamFrame]:
        """Iterate over selected channels while preserving global order."""
        return self.subscribe(channels=channels)

    def subscribe(
        self, channels: Sequence[StreamChannel] | None = None
    ) -> Iterator[StreamFrame]:
        """Iterate over frames, optionally filtered by channel."""
        if self._sync_iterator is None:
            raise RuntimeError("Sync iterator not available")
        selected = set(channels) if channels is not None else None
        try:
            for frame in self._sync_iterator:
                self._frames.append(frame)
                if selected is None or frame.channel in selected:
                    yield frame
            self._exhausted = True
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def close(self) -> None:
        """Cancel streaming and clean up resources."""
        if self._cancelled or self._exhausted or self._error is not None:
            return
        self._cancelled = True
        self._completed = True
        if self._sync_iterator is not None and hasattr(self._sync_iterator, "close"):
            self._sync_iterator.close()
        if self._on_cleanup is not None:
            self._on_cleanup()
            self._on_cleanup = None


class AsyncStreamSession(StreamSessionBase[T]):
    """Asynchronous stream session for ordered public frames."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if not self._exhausted:
            await self.aclose()

    @property
    def events(self) -> AsyncIterator[StreamFrame]:
        """Iterate over all ordered frames."""
        return self.subscribe()

    @property
    def llm(self) -> AsyncIterator[StreamFrame]:
        """Iterate over LLM token and thinking frames."""
        return self.subscribe(channels=["llm"])

    @property
    def messages(self) -> AsyncIterator[StreamFrame]:
        """Iterate over conversation message frames."""
        return self.subscribe(channels=["messages"])

    @property
    def flow(self) -> AsyncIterator[StreamFrame]:
        """Iterate over Flow lifecycle and method frames."""
        return self.subscribe(channels=["flow"])

    @property
    def tools(self) -> AsyncIterator[StreamFrame]:
        """Iterate over tool execution frames."""
        return self.subscribe(channels=["tools"])

    def interleave(
        self, channels: Sequence[StreamChannel]
    ) -> AsyncIterator[StreamFrame]:
        """Iterate over selected channels while preserving global order."""
        return self.subscribe(channels=channels)

    async def subscribe(
        self, channels: Sequence[StreamChannel] | None = None
    ) -> AsyncIterator[StreamFrame]:
        """Iterate over frames, optionally filtered by channel."""
        if self._async_iterator is None:
            raise RuntimeError("Async iterator not available")
        selected = set(channels) if channels is not None else None
        try:
            async for frame in self._async_iterator:
                self._frames.append(frame)
                if selected is None or frame.channel in selected:
                    yield frame
            self._exhausted = True
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    async def aclose(self) -> None:
        """Cancel streaming and clean up resources."""
        if self._cancelled or self._exhausted or self._error is not None:
            return
        self._cancelled = True
        self._completed = True
        if self._async_iterator is not None and hasattr(self._async_iterator, "aclose"):
            await self._async_iterator.aclose()
        if self._on_cleanup is not None:
            self._on_cleanup()
            self._on_cleanup = None


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
    """

    def __init__(
        self,
        sync_iterator: Iterator[StreamChunk] | None = None,
        async_iterator: AsyncIterator[StreamChunk] | None = None,
    ) -> None:
        """Initialize streaming output base."""
        self._result: T | None = None
        self._completed: bool = False
        self._chunks: list[StreamChunk] = []
        self._error: Exception | None = None
        self._cancelled: bool = False
        self._exhausted: bool = False
        self._on_cleanup: Callable[[], None] | None = None
        self._sync_iterator = sync_iterator
        self._async_iterator = async_iterator

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

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Exit async context manager, cancelling if still running."""
        await self.aclose()

    async def aclose(self) -> None:
        """Cancel streaming and clean up resources.

        Cancels any in-flight tasks and closes the underlying async iterator.
        Safe to call multiple times. No-op if already cancelled or fully consumed.
        """
        if self._cancelled or self._exhausted or self._error is not None:
            return
        self._cancelled = True
        self._completed = True
        if self._async_iterator is not None and hasattr(self._async_iterator, "aclose"):
            await self._async_iterator.aclose()
        if self._on_cleanup is not None:
            self._on_cleanup()
            self._on_cleanup = None

    def close(self) -> None:
        """Cancel streaming and clean up resources (sync).

        Closes the underlying sync iterator. Safe to call multiple times.
        No-op if already cancelled, fully consumed, or errored.
        """
        if self._cancelled or self._exhausted or self._error is not None:
            return
        self._cancelled = True
        self._completed = True
        if self._sync_iterator is not None and hasattr(self._sync_iterator, "close"):
            self._sync_iterator.close()
        if self._on_cleanup is not None:
            self._on_cleanup()
            self._on_cleanup = None

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
            self._exhausted = True
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
            self._exhausted = True
        except Exception as e:
            self._error = e
            raise
        finally:
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
        super().__init__(sync_iterator=sync_iterator, async_iterator=async_iterator)
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

    def _set_result(self, result: Any) -> None:
        """Set the final result after streaming completes.

        Args:
            result: The final flow output.
        """
        self._result = result
        self._completed = True


class LLMStreamingOutput(StreamingOutputBase[Any]):
    """Streaming output wrapper for direct LLM calls."""

    def _set_result(self, result: Any) -> None:
        """Set the final LLM call result after streaming completes."""
        self._result = result
        self._completed = True
