"""Queue-backed input provider for conversational flows."""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


class QueueInputProvider:
    """Blocks on a per-session queue until a user message is pushed.

    Use for long-running workers where ``Flow.ask()`` should wait on WebSocket
    or another transport without blocking the event loop thread (flow runs ask
    in a worker thread).

    Example:
        ```python
        provider = QueueInputProvider()
        flow.input_provider = provider

        # From a WebSocket handler:
        provider.push(session_id, "hello")

        # Inside the flow:
        reply = flow.ask("You: ", metadata={"session_id": session_id})
        ```
    """

    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue[str | None]] = {}
        self._lock = threading.Lock()

    def _get_queue(self, session_id: str) -> queue.Queue[str | None]:
        with self._lock:
            if session_id not in self._queues:
                self._queues[session_id] = queue.Queue()
            return self._queues[session_id]

    def push(self, session_id: str, text: str) -> None:
        """Enqueue a user message for the given session."""
        self._get_queue(session_id).put(text)

    def close_session(self, session_id: str) -> None:
        """Signal end of session (unblocks ``ask()`` with None)."""
        self._get_queue(session_id).put(None)

    def request_input(
        self,
        message: str,
        flow: Flow[Any],
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        session_id = self._resolve_session_id(flow, metadata)
        if session_id is None:
            return None
        try:
            return self._get_queue(session_id).get()
        except Exception:
            return None

    @staticmethod
    def _resolve_session_id(
        flow: Flow[Any],
        metadata: dict[str, Any] | None,
    ) -> str | None:
        if metadata and metadata.get("session_id"):
            return str(metadata["session_id"])
        state = getattr(flow, "_state", None)
        if state is None:
            return None
        if isinstance(state, dict):
            value = state.get("id")
            return str(value) if value else None
        if hasattr(state, "id"):
            value = getattr(state, "id", None)
            return str(value) if value else None
        return None
