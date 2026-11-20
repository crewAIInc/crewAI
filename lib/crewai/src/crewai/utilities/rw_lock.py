"""Read-write lock for thread-safe concurrent access.

This module provides a reader-writer lock implementation that allows multiple
concurrent readers or a single exclusive writer.
"""

from collections.abc import Generator
from contextlib import contextmanager
from threading import Condition, local


class RWLock:
    """Read-write lock for managing concurrent read and exclusive write access.

    Allows multiple threads to acquire read locks simultaneously, but ensures
    exclusive access for write operations. Writers are prioritized when waiting.

    Attributes:
        _cond: Condition variable for coordinating lock access
        _readers: Count of active readers
        _writer: Whether a writer currently holds the lock
    """

    def __init__(self) -> None:
        """Initialize the read-write lock."""
        self._cond = Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0
        self._read_depth = local()

    def r_acquire(self) -> None:
        """Acquire a read lock, blocking if a writer holds or awaits the lock."""
        depth = getattr(self._read_depth, "value", 0)
        with self._cond:
            # A reader that already holds the lock re-enters without waiting:
            # blocking it here would deadlock the thread against its own read
            # lock once a writer queues up.
            if depth == 0:
                while self._writer or self._waiting_writers > 0:
                    self._cond.wait()
            self._readers += 1
        self._read_depth.value = depth + 1

    def r_release(self) -> None:
        """Release a read lock and notify waiting writers if last reader."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()
        self._read_depth.value = max(getattr(self._read_depth, "value", 1) - 1, 0)

    @contextmanager
    def r_locked(self) -> Generator[None, None, None]:
        """Context manager for acquiring a read lock.

        Yields:
            None
        """
        self.r_acquire()
        try:
            yield
        finally:
            self.r_release()

    def w_acquire(self) -> None:
        """Acquire a write lock, blocking if any readers or writers are active."""
        with self._cond:
            # Readers that arrive from now on wait for this writer, otherwise a
            # steady stream of readers starves it forever.
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    def w_release(self) -> None:
        """Release a write lock and notify all waiting threads."""
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    @contextmanager
    def w_locked(self) -> Generator[None, None, None]:
        """Context manager for acquiring a write lock.

        Yields:
            None
        """
        self.w_acquire()
        try:
            yield
        finally:
            self.w_release()
