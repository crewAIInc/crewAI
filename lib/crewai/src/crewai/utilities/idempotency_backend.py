"""Pluggable idempotency backends for tool execution deduplication.

Provides an abstract interface and a default in-memory implementation.
Users requiring cross-process / cross-worker durability can inject a
custom backend (e.g. Redis, database) via :meth:`ToolsHandler.set_idempotency_backend`.

The in-memory default is *process-local* — it will NOT prevent duplicate
tool side effects when a retry is dispatched to a different worker process.
For multi-worker deployments, use a persistent backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock
from typing import Final


# Sentinel stored by claim() to mark an in-progress execution.
_IN_PROGRESS: Final[str] = "__idempotency_in_progress__"


class IdempotencyBackend(ABC):
    """Abstract interface for an idempotency store.

    Implementations must be thread-safe.  Cross-process safety is
    implementation-dependent — the default :class:`MemoryIdempotencyBackend`
    is process-local only.
    """

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Return a previously stored result, or None."""
        ...

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Store a result for the given key."""
        ...

    @abstractmethod
    def claim(self, key: str) -> tuple[bool, str | None]:
        """Atomically try to claim execution rights for *key*.

        Returns a ``(claimed, result)`` tuple:

        * ``(True, None)`` — the caller owns the claim and **must** execute
          the tool, then call :meth:`set` to publish the result.
        * ``(False, result)`` — another caller already claimed or completed
          this key.  If *result* is not ``None`` it is the final result that
          can be returned immediately.  If *result* is ``None`` the other
          caller is still in progress; the current caller should wait/retry
          or fall back to normal execution.
        """
        ...

    @abstractmethod
    def release(self, key: str) -> None:
        """Release an in-progress claim without publishing a result.

        Call this when the caller that won the claim fails before calling
        :meth:`set`, so that subsequent callers are not blocked by a stale
        in-progress marker.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored entries (e.g. at the start of a fresh task)."""
        ...


class MemoryIdempotencyBackend(IdempotencyBackend):
    """In-memory idempotency store (process-local, thread-safe).

    .. warning::
        This backend lives in the process memory of a single worker.
        It **will not** deduplicate tool executions across worker
        processes or restarts.  For multi-worker deployments, inject a
        persistent :class:`IdempotencyBackend` backed by Redis or a database.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._lock = Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            value = self._store.get(key)
            if value == _IN_PROGRESS:
                return None  # treat in-progress as "not yet available"
            return value

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._store[key] = value

    def claim(self, key: str) -> tuple[bool, str | None]:
        with self._lock:
            existing = self._store.get(key)
            if existing is not None:
                if existing == _IN_PROGRESS:
                    # Another caller is still executing — tell caller to
                    # wait or proceed normally.
                    return (False, None)
                # Already completed — return the stored result.
                return (False, existing)
            # We own the claim.
            self._store[key] = _IN_PROGRESS
            return (True, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def release(self, key: str) -> None:
        """Remove the in-progress marker for *key* without storing a result.

        Only clears the entry if it is currently ``_IN_PROGRESS`` — completed
        results are left untouched to preserve idempotency.
        """
        with self._lock:
            if self._store.get(key) == _IN_PROGRESS:
                del self._store[key]
