"""Centralised lock factory.

If ``REDIS_URL`` is set, locks are distributed via ``portalocker.RedisLock``. Otherwise, falls
back to the standard ``portalocker.Lock``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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


@lru_cache(maxsize=1)
def _redis_connection() -> redis.Redis:
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
    channel = f"crewai:{md5(name.encode(), usedforsecurity=False).hexdigest()}"

    if _REDIS_URL:
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
