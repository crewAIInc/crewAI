"""Internal FIFO queue for conversational Flow turns."""

from __future__ import annotations

from concurrent.futures import Future
from contextvars import Context, copy_context
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread, current_thread
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


_FLOW_QUEUE_ATTR = "_conversation_turn_queue"
_FLOW_QUEUE_LOCK = Lock()
_STOP = object()


def _track_feature(feature: str) -> None:
    """Count a queue feature without letting telemetry break the session."""
    try:
        from crewai.telemetry.telemetry import Telemetry

        Telemetry().feature_usage_span(feature)
    except Exception:
        return


class ConversationTurnQueueError(RuntimeError):
    """Base error raised by the conversation turn queue."""


class ConversationTurnQueueFullError(ConversationTurnQueueError):
    """Raised when a queue has reached its pending-turn limit."""


class ConversationTurnQueueStoppedError(ConversationTurnQueueError):
    """Raised when work is submitted to a closed or failed queue."""


@dataclass(frozen=True)
class _QueuedTurn:
    message: str
    kwargs: dict[str, Any]
    future: Future[Any]
    context: Context


class ConversationTurnQueue:
    """Run submitted conversational turns in FIFO order on one worker.

    The queue is runtime-only. It serializes calls to ``Flow.handle_turn``
    without adding pending messages to persisted conversation history before
    their turn begins.
    """

    def __init__(
        self,
        flow: Flow[Any],
        *,
        session_id: str | None = None,
        max_pending: int = 32,
        handle_turn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")

        self._flow = flow
        self._session_id = session_id
        self._max_pending = max_pending
        self._handle_turn_kwargs = dict(handle_turn_kwargs or {})
        self._items: Queue[_QueuedTurn | object] = Queue(maxsize=max_pending)
        self._state_lock = Lock()
        self._closed = False
        self._active = False
        self._pending = 0
        self._failure: BaseException | None = None
        self._worker = Thread(
            target=self._run,
            name=f"{type(flow).__name__}-conversation-turns",
        )

        self._claim_flow()
        try:
            self._worker.start()
        except BaseException:
            self._release_flow()
            raise

    @property
    def active(self) -> bool:
        """Whether one turn is currently executing."""
        with self._state_lock:
            return self._active

    @property
    def pending_count(self) -> int:
        """Number of accepted turns waiting behind the active turn."""
        with self._state_lock:
            return self._pending

    @property
    def closed(self) -> bool:
        """Whether the queue has stopped accepting submissions."""
        with self._state_lock:
            return self._closed

    @property
    def failure(self) -> BaseException | None:
        """The turn error that stopped the queue, if any."""
        with self._state_lock:
            return self._failure

    def submit(self, message: str, **handle_turn_kwargs: Any) -> Future[Any]:
        """Queue one message and immediately return its result future."""
        future: Future[Any] = Future()
        item = _QueuedTurn(
            message=message,
            kwargs={**self._handle_turn_kwargs, **handle_turn_kwargs},
            future=future,
            context=copy_context(),
        )

        with self._state_lock:
            if self._closed:
                raise ConversationTurnQueueStoppedError(
                    "Conversation turn queue is closed"
                )
            backed_up = self._active or self._pending > 0
            try:
                self._items.put_nowait(item)
            except Full as exc:
                _track_feature("flow:conversation_queue_full")
                raise ConversationTurnQueueFullError(
                    f"Conversation turn queue already has {self._max_pending} "
                    "pending turns"
                ) from exc
            self._pending += 1
        if backed_up:
            _track_feature("flow:conversation_queued")
        return future

    def close(self, *, wait: bool = True) -> None:
        """Stop accepting submissions and drain accepted turns."""
        if wait and current_thread() is self._worker:
            raise ConversationTurnQueueError(
                "Cannot wait for the conversation queue from its worker thread"
            )

        with self._state_lock:
            should_signal = not self._closed
            self._closed = True
        if should_signal:
            self._items.put(_STOP)
        if wait:
            self._worker.join()

    def __enter__(self) -> ConversationTurnQueue:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _claim_flow(self) -> None:
        with _FLOW_QUEUE_LOCK:
            existing = getattr(self._flow, _FLOW_QUEUE_ATTR, None)
            if existing is not None:
                raise ConversationTurnQueueError(
                    "This Flow already has an active conversation turn queue"
                )
            object.__setattr__(self._flow, _FLOW_QUEUE_ATTR, self)

    def _release_flow(self) -> None:
        with _FLOW_QUEUE_LOCK:
            if getattr(self._flow, _FLOW_QUEUE_ATTR, None) is self:
                object.__setattr__(self._flow, _FLOW_QUEUE_ATTR, None)

    def _run(self) -> None:
        try:
            while True:
                queued = self._items.get()
                if not isinstance(queued, _QueuedTurn):
                    return

                turn = queued
                with self._state_lock:
                    self._pending -= 1
                    self._active = True
                try:
                    if turn.future.set_running_or_notify_cancel():
                        kwargs = dict(turn.kwargs)
                        if self._session_id is not None:
                            kwargs["session_id"] = self._session_id
                        result = turn.context.run(
                            self._flow.handle_turn,
                            turn.message,
                            **kwargs,
                        )
                        turn.future.set_result(result)
                except BaseException as exc:
                    turn.future.set_exception(exc)
                    self._stop_after_failure(exc)
                    return
                finally:
                    with self._state_lock:
                        self._active = False
        finally:
            self._release_flow()

    def _stop_after_failure(self, failure: BaseException) -> None:
        with self._state_lock:
            self._failure = failure
            self._closed = True

        dropped = False
        while True:
            try:
                queued = self._items.get_nowait()
            except Empty:
                break
            if not isinstance(queued, _QueuedTurn):
                continue
            dropped = True
            with self._state_lock:
                self._pending -= 1
            if not queued.future.cancelled():
                queued.future.set_exception(
                    ConversationTurnQueueStoppedError(
                        "Conversation turn queue stopped after an earlier turn failed"
                    )
                )
        if dropped:
            _track_feature("flow:conversation_queue_dropped")


__all__ = [
    "ConversationTurnQueue",
    "ConversationTurnQueueError",
    "ConversationTurnQueueFullError",
    "ConversationTurnQueueStoppedError",
]
