"""Read-write lock for thread-safe concurrent access.

This module provides a reader-writer lock implementation that allows multiple
concurrent readers or a single exclusive writer.
"""

from collections.abc import Generator
from contextlib import contextmanager
from threading import Condition


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

    def r_acquire(self) -> None:
        """Acquire a read lock, blocking if a writer holds the lock."""
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def r_release(self) -> None:
        """Release a read lock and notify waiting writers if last reader."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    @contextmanager
    def r_locked(self) -> Generator[None, None, None]:
        """Context manager for acquiring a read lock.

        Yields:
            None
        """
        try:
            self.r_acquire()
            yield
        finally:
            self.r_release()

    def w_acquire(self) -> None:
        """Acquire a write lock, blocking if any readers or writers are active."""
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

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
        try:
            self.w_acquire()
            yield
        finally:
            self.w_release()
