"""Centralised lock factory.

By default, if ``REDIS_URL`` is set and the ``redis`` package is installed,
locks are distributed via ``portalocker.RedisLock``. Otherwise, falls back to
the standard file-based ``portalocker.Lock`` in the system temp dir.

The backend can be replaced via :func:`set_lock_backend` to plug in a custom
locking strategy (e.g. a different distributed lock service, or an in-process
lock for tests).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import lru_cache
from hashlib import md5
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Final

import portalocker
import portalocker.exceptions


if TYPE_CHECKING:
    import redis


logger = logging.getLogger(__name__)

_REDIS_URL: str | None = os.environ.get("REDIS_URL")

_DEFAULT_TIMEOUT: Final[int] = 120

# A backend is called as ``backend(name, timeout=...)`` and returns a context
# manager that holds the lock while the ``with`` block runs.
LockBackend = Callable[..., AbstractContextManager[None]]

# ``None`` means use the built-in Redis/file selection.
_backend: LockBackend | None = None


def set_lock_backend(backend: LockBackend | None) -> None:
    """Replace the locking backend used by :func:`lock`.

    Pass ``None`` to restore the built-in Redis/file default.
    """
    global _backend
    _backend = backend


def _redis_available() -> bool:
    """Return True if redis is installed and REDIS_URL is set."""
    if not _REDIS_URL:
        return False
    try:
        import redis  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _redis_connection() -> redis.Redis[bytes]:
    """Return a cached Redis connection, creating one on first call."""
    from redis import Redis

    if _REDIS_URL is None:
        raise ValueError("REDIS_URL environment variable is not set")
    return Redis.from_url(_REDIS_URL)


@contextmanager
def lock(name: str, *, timeout: float = _DEFAULT_TIMEOUT) -> Iterator[None]:
    """Acquire a named lock, yielding while it is held.

    Args:
        name: A human-readable lock name (e.g. ``"chromadb_init"``).
              Automatically namespaced to avoid collisions.
        timeout: Maximum seconds to wait for the lock before raising.
    """
    # Snapshot the global once: a concurrent set_lock_backend() must not turn
    # the check-then-call into calling ``None``.
    backend = _backend
    if backend is not None:
        with backend(name, timeout=timeout):
            yield
        return

    channel = f"crewai:{md5(name.encode(), usedforsecurity=False).hexdigest()}"

    if _redis_available():
        with portalocker.RedisLock(
            channel=channel,
            connection=_redis_connection(),
            timeout=timeout,
        ):
            yield
    else:
        lock_dir = tempfile.gettempdir()
        lock_path = os.path.join(lock_dir, f"{channel}.lock")
        try:
            pl = portalocker.Lock(lock_path, timeout=timeout)
            pl.acquire()
        except portalocker.exceptions.BaseLockException as exc:
            raise portalocker.exceptions.LockException(
                f"Failed to acquire lock '{name}' at {lock_path} "
                f"(timeout={timeout}s). This commonly occurs in "
                f"multi-process environments. "
            ) from exc
        try:
            yield
        finally:
            pl.release()  # type: ignore[no-untyped-call]
