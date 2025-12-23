"""WebSocket and SSE types for continuous streaming."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from crewai.types.continuous_streaming import ContinuousStreamChunk


class WebSocketMessage(BaseModel):
    """Standard WebSocket message format for continuous streaming.

    This provides a consistent message format that can be easily
    consumed by WebSocket clients and SSE endpoints.

    Attributes:
        type: Message type (chunk, action, observation, error, heartbeat)
        data: Message payload data
        timestamp: ISO formatted timestamp
        iteration: Continuous iteration number
    """

    type: str = Field(description="Message type")
    data: dict[str, Any] = Field(default_factory=dict, description="Message payload")
    timestamp: str = Field(description="ISO formatted timestamp")
    iteration: int = Field(default=0, description="Continuous iteration number")

    @classmethod
    def from_chunk(cls, chunk: ContinuousStreamChunk) -> "WebSocketMessage":
        """Create WebSocket message from a continuous stream chunk.

        Args:
            chunk: The stream chunk to convert

        Returns:
            WebSocketMessage suitable for WebSocket transmission
        """
        data: dict[str, Any] = {
            "content": chunk.content,
            "agent_role": chunk.agent_role,
            "agent_id": chunk.agent_id,
        }

        if chunk.tool_call:
            data["tool_call"] = {
                "tool_id": chunk.tool_call.tool_id,
                "tool_name": chunk.tool_call.tool_name,
                "arguments": chunk.tool_call.arguments,
            }

        return cls(
            type=chunk.chunk_type.value,
            data=data,
            timestamp=chunk.timestamp.isoformat(),
            iteration=chunk.iteration,
        )

    @classmethod
    def error(cls, error: str, iteration: int = 0) -> "WebSocketMessage":
        """Create an error message.

        Args:
            error: Error description
            iteration: Current iteration number

        Returns:
            Error WebSocketMessage
        """
        return cls(
            type="error",
            data={"error": error},
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
        )

    @classmethod
    def heartbeat(cls, iteration: int = 0) -> "WebSocketMessage":
        """Create a heartbeat message.

        Args:
            iteration: Current iteration number

        Returns:
            Heartbeat WebSocketMessage
        """
        return cls(
            type="heartbeat",
            data={},
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
        )

    @classmethod
    def status(
        cls,
        status: str,
        iteration: int = 0,
        **extra: Any
    ) -> "WebSocketMessage":
        """Create a status message.

        Args:
            status: Status string (started, stopped, paused, resumed)
            iteration: Current iteration number
            **extra: Additional data to include

        Returns:
            Status WebSocketMessage
        """
        return cls(
            type="status",
            data={"status": status, **extra},
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
        )

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation
        """
        return self.model_dump_json()


def chunk_to_sse(chunk: ContinuousStreamChunk) -> str:
    """Convert a continuous stream chunk to SSE format.

    Args:
        chunk: The stream chunk to convert

    Returns:
        SSE formatted string ready for streaming
    """
    return f"data: {chunk.to_websocket_message()}\n\n"


def message_to_sse(message: WebSocketMessage) -> str:
    """Convert a WebSocket message to SSE format.

    Args:
        message: The message to convert

    Returns:
        SSE formatted string ready for streaming
    """
    return f"data: {message.to_json()}\n\n"


class SSEFormatter:
    """Helper class for formatting SSE streams.

    This class provides utilities for properly formatting
    Server-Sent Events streams with support for event types
    and retry configuration.

    Example:
        ```python
        formatter = SSEFormatter()

        # Format a chunk
        sse_data = formatter.format_chunk(chunk)

        # Format with event type
        sse_data = formatter.format_event("action", {"tool": "search"})

        # Format retry instruction
        sse_data = formatter.retry(3000)
        ```
    """

    @staticmethod
    def format_chunk(chunk: ContinuousStreamChunk) -> str:
        """Format a chunk as SSE data.

        Args:
            chunk: The chunk to format

        Returns:
            SSE formatted string
        """
        return chunk_to_sse(chunk)

    @staticmethod
    def format_message(message: WebSocketMessage) -> str:
        """Format a WebSocket message as SSE data.

        Args:
            message: The message to format

        Returns:
            SSE formatted string
        """
        return message_to_sse(message)

    @staticmethod
    def format_event(
        event_type: str,
        data: dict[str, Any],
        iteration: int = 0
    ) -> str:
        """Format an event with specific type.

        Args:
            event_type: The event type (appears as 'event:' line)
            data: Event data
            iteration: Iteration number

        Returns:
            SSE formatted string with event type
        """
        message = WebSocketMessage(
            type=event_type,
            data=data,
            timestamp=datetime.now().isoformat(),
            iteration=iteration,
        )
        return f"event: {event_type}\ndata: {message.to_json()}\n\n"

    @staticmethod
    def retry(milliseconds: int) -> str:
        """Format a retry instruction.

        Args:
            milliseconds: Retry interval in milliseconds

        Returns:
            SSE retry line
        """
        return f"retry: {milliseconds}\n\n"

    @staticmethod
    def comment(text: str) -> str:
        """Format a comment (often used as keep-alive).

        Args:
            text: Comment text

        Returns:
            SSE comment line
        """
        return f": {text}\n\n"

    @staticmethod
    def keep_alive() -> str:
        """Generate a keep-alive comment.

        Returns:
            SSE keep-alive comment
        """
        return ": keep-alive\n\n"
