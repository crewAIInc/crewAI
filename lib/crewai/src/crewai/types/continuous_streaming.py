"""Streaming output types for continuous operation mode."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from crewai.types.streaming import StreamChunkType, ToolCallChunk

if TYPE_CHECKING:
    from crewai.continuous.shutdown import ShutdownController


class ContinuousChunkType(str, Enum):
    """Type of continuous streaming chunk."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    ACTION = "action"
    OBSERVATION = "observation"
    CHECKPOINT = "checkpoint"
    HEARTBEAT = "heartbeat"


class ContinuousStreamChunk(BaseModel):
    """Streaming chunk for continuous operation mode.

    Attributes:
        content: The streaming content (text or partial content)
        chunk_type: Type of the chunk (text, tool_call, action, etc.)
        iteration: Which continuous iteration this chunk belongs to
        timestamp: When this chunk was generated
        agent_role: Role of the agent generating this chunk
        agent_id: Unique identifier of the agent
        tool_call: Tool call information if applicable
    """

    content: str = Field(description="The streaming content")
    chunk_type: ContinuousChunkType = Field(
        default=ContinuousChunkType.TEXT, description="Type of the chunk"
    )
    iteration: int = Field(default=0, description="Continuous iteration number")
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_role: str = Field(default="", description="Role of the agent")
    agent_id: str = Field(default="", description="Unique identifier of the agent")
    tool_call: ToolCallChunk | None = Field(
        default=None, description="Tool call information"
    )

    def __str__(self) -> str:
        """Return the chunk content as a string."""
        return self.content

    def to_websocket_message(self) -> str:
        """Convert chunk to WebSocket-ready JSON string.

        Returns:
            JSON string suitable for WebSocket transmission
        """
        return self.model_dump_json()

    def to_sse(self) -> str:
        """Convert chunk to Server-Sent Events format.

        Returns:
            SSE formatted string
        """
        return f"data: {self.to_websocket_message()}\n\n"


class ContinuousStreamingOutput:
    """Output handler for continuous streaming.

    Provides multiple consumption methods:
    1. Async generator: `async for chunk in output`
    2. Sync generator: `for chunk in output`
    3. Callback registration: `output.on_chunk(handler)`

    This class manages the stream lifecycle and ensures chunks are
    delivered to all registered consumers.

    Example:
        ```python
        # Method 1: Async generator
        async for chunk in output:
            print(chunk.content, end="", flush=True)

        # Method 2: Sync generator
        for chunk in output:
            print(chunk.content, end="", flush=True)

        # Method 3: Callbacks
        output.on_chunk(lambda c: print(c.content, end=""))
        ```
    """

    def __init__(self, shutdown_controller: "ShutdownController") -> None:
        """Initialize continuous streaming output.

        Args:
            shutdown_controller: Controller for shutdown coordination
        """
        self._shutdown = shutdown_controller
        self._sync_queue: queue.Queue[ContinuousStreamChunk | None] = queue.Queue()
        self._async_queue: asyncio.Queue[ContinuousStreamChunk | None] | None = None
        self._callbacks: list[Callable[[ContinuousStreamChunk], None]] = []
        self._chunks: list[ContinuousStreamChunk] = []
        self._lock = threading.Lock()
        self._current_iteration = 0
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the streaming output.

        Args:
            loop: Optional event loop for async queue operations
        """
        self._started = True
        self._loop = loop
        if loop is not None:
            self._async_queue = asyncio.Queue()

    def stop(self) -> None:
        """Stop the streaming output and signal completion to consumers."""
        # Signal end to sync queue
        self._sync_queue.put(None)

        # Signal end to async queue
        if self._async_queue is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._async_queue.put_nowait, None
                )
            except RuntimeError:
                # Loop might be closed
                pass

    @property
    def is_started(self) -> bool:
        """Check if streaming has started."""
        return self._started

    @property
    def current_iteration(self) -> int:
        """Get current iteration number."""
        return self._current_iteration

    @current_iteration.setter
    def current_iteration(self, value: int) -> None:
        """Set current iteration number."""
        self._current_iteration = value

    @property
    def chunks(self) -> list[ContinuousStreamChunk]:
        """Get all collected chunks."""
        with self._lock:
            return self._chunks.copy()

    def on_chunk(self, callback: Callable[[ContinuousStreamChunk], None]) -> None:
        """Register a callback to be called for each chunk.

        Args:
            callback: Function to call with each chunk
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ContinuousStreamChunk], None]) -> None:
        """Remove a registered callback.

        Args:
            callback: The callback to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def emit(self, chunk: ContinuousStreamChunk) -> None:
        """Emit a chunk to all consumers.

        Args:
            chunk: The chunk to emit
        """
        if self._shutdown.should_stop:
            return

        with self._lock:
            self._chunks.append(chunk)
            callbacks = self._callbacks.copy()

        # Send to sync queue
        self._sync_queue.put(chunk)

        # Send to async queue
        if self._async_queue is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    self._async_queue.put_nowait, chunk
                )
            except RuntimeError:
                # Loop might be closed
                pass

        # Call registered callbacks
        for callback in callbacks:
            try:
                callback(chunk)
            except Exception:
                # Don't let callback errors break streaming
                pass

    def emit_text(
        self,
        content: str,
        agent_role: str = "",
        agent_id: str = "",
    ) -> None:
        """Convenience method to emit a text chunk.

        Args:
            content: Text content to emit
            agent_role: Role of the agent
            agent_id: ID of the agent
        """
        chunk = ContinuousStreamChunk(
            content=content,
            chunk_type=ContinuousChunkType.TEXT,
            iteration=self._current_iteration,
            timestamp=datetime.now(),
            agent_role=agent_role,
            agent_id=agent_id,
        )
        self.emit(chunk)

    def emit_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        agent_role: str = "",
        agent_id: str = "",
    ) -> None:
        """Convenience method to emit a tool call chunk.

        Args:
            tool_name: Name of the tool being called
            tool_input: Arguments for the tool
            agent_role: Role of the agent
            agent_id: ID of the agent
        """
        import json

        chunk = ContinuousStreamChunk(
            content=f"Calling tool: {tool_name}",
            chunk_type=ContinuousChunkType.TOOL_CALL,
            iteration=self._current_iteration,
            timestamp=datetime.now(),
            agent_role=agent_role,
            agent_id=agent_id,
            tool_call=ToolCallChunk(
                tool_name=tool_name,
                arguments=json.dumps(tool_input) if tool_input else "",
            ),
        )
        self.emit(chunk)

    def emit_observation(
        self,
        observation: str,
        agent_role: str = "",
        agent_id: str = "",
    ) -> None:
        """Convenience method to emit an observation chunk.

        Args:
            observation: The observation content
            agent_role: Role of the agent
            agent_id: ID of the agent
        """
        chunk = ContinuousStreamChunk(
            content=observation,
            chunk_type=ContinuousChunkType.OBSERVATION,
            iteration=self._current_iteration,
            timestamp=datetime.now(),
            agent_role=agent_role,
            agent_id=agent_id,
        )
        self.emit(chunk)

    def emit_heartbeat(self) -> None:
        """Emit a heartbeat chunk to keep connection alive."""
        chunk = ContinuousStreamChunk(
            content="",
            chunk_type=ContinuousChunkType.HEARTBEAT,
            iteration=self._current_iteration,
            timestamp=datetime.now(),
        )
        self.emit(chunk)

    def __iter__(self) -> Iterator[ContinuousStreamChunk]:
        """Synchronous iterator for streaming chunks.

        Yields:
            ContinuousStreamChunk objects as they arrive.
        """
        while not self._shutdown.should_stop:
            try:
                chunk = self._sync_queue.get(timeout=1.0)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                continue

    async def __aiter__(self) -> AsyncIterator[ContinuousStreamChunk]:
        """Asynchronous iterator for streaming chunks.

        Yields:
            ContinuousStreamChunk objects as they arrive.
        """
        if self._async_queue is None:
            # Fallback to sync queue with async wrapper
            while not self._shutdown.should_stop:
                try:
                    chunk = self._sync_queue.get_nowait()
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    await asyncio.sleep(0.01)
            return

        while not self._shutdown.should_stop:
            try:
                chunk = await asyncio.wait_for(
                    self._async_queue.get(),
                    timeout=1.0
                )
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                continue

    def get_full_text(self) -> str:
        """Get all text content concatenated.

        Returns:
            All text chunks concatenated together.
        """
        with self._lock:
            return "".join(
                chunk.content
                for chunk in self._chunks
                if chunk.chunk_type == ContinuousChunkType.TEXT
            )

    def get_stats(self) -> dict[str, Any]:
        """Get streaming statistics.

        Returns:
            Dictionary with streaming statistics
        """
        with self._lock:
            return {
                "total_chunks": len(self._chunks),
                "current_iteration": self._current_iteration,
                "callbacks_registered": len(self._callbacks),
                "is_started": self._started,
            }
