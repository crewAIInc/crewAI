"""Centralised lock factory.

The locking backend is resolved in this order of precedence:

1. A backend registered in-process via :func:`set_lock_backend`. Best for
   tests and runtime wiring.
2. A backend named by the ``CREWAI_LOCK_FACTORY`` environment variable, in
   ``"module:callable"`` form (e.g. ``"my_pkg.locks:lock"``). The import path
   is resolved lazily and cached. Best for deployment-driven selection, since
   it requires no code changes and rolls back with an env unset.
3. The built-in default: if ``REDIS_URL`` is set and the ``redis`` package is
   installed, locks are distributed via ``portalocker.RedisLock``; otherwise
   they fall back to a file-based ``portalocker.Lock`` in the system temp dir.

A custom backend is any callable matching :class:`LockBackend`. It receives the
raw lock ``name`` (not the ``crewai:<hash>`` channel) and owns its own
namespacing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from hashlib import md5
import importlib
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import portalocker
import portalocker.exceptions


if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    import redis


logger = logging.getLogger(__name__)

_REDIS_URL: str | None = os.environ.get("REDIS_URL")

# Optional "module:callable" import path for a custom lock backend. Read once at
# import time, mirroring ``_REDIS_URL``; the env must be set before the process
# starts.
_LOCK_FACTORY_SPEC: str | None = os.environ.get("CREWAI_LOCK_FACTORY")

_DEFAULT_TIMEOUT: Final[int] = 120


@runtime_checkable
class LockBackend(Protocol):
    """A pluggable locking backend.

    A backend is any callable that, given a raw lock ``name`` and a
    ``timeout``, returns a context manager that holds the lock for the
    duration of the ``with`` block and releases it on exit. The ``name`` is
    passed through verbatim (e.g. ``"chromadb_init"``); the backend owns its
    own namespacing.
    """

    def __call__(
        self, name: str, *, timeout: float
    ) -> AbstractContextManager[None]:
        raise NotImplementedError


# Active backend override; ``None`` means use the built-in default selection.
_backend: LockBackend | None = None


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


@lru_cache(maxsize=1)
def _env_lock_factory() -> LockBackend | None:
    """Resolve the ``CREWAI_LOCK_FACTORY`` import path to a callable.

    Returns ``None`` when the env var is unset. Resolution is cached, so the
    import happens at most once per process.

    Raises:
        ValueError: if the spec is not in ``"module:callable"`` form.
        ImportError / AttributeError: if the module or attribute is missing.
        TypeError: if the resolved attribute is not callable.
    """
    if not _LOCK_FACTORY_SPEC:
        return None

    module_path, sep, attr = _LOCK_FACTORY_SPEC.partition(":")
    if not sep or not module_path or not attr:
        raise ValueError(
            "CREWAI_LOCK_FACTORY must be in 'module:callable' form, "
            f"got {_LOCK_FACTORY_SPEC!r}"
        )

    module = importlib.import_module(module_path)
    factory: LockBackend = getattr(module, attr)
    if not callable(factory):
        raise TypeError(
            f"CREWAI_LOCK_FACTORY={_LOCK_FACTORY_SPEC!r} resolved to a "
            f"non-callable {type(factory).__name__}; expected a callable "
            "matching LockBackend (name, *, timeout) -> context manager."
        )
    logger.debug("Using custom lock backend from %s", _LOCK_FACTORY_SPEC)
    return factory


def _active_backend() -> LockBackend:
    """Return the backend to use, honouring override > env > default."""
    if _backend is not None:
        return _backend
    env_factory = _env_lock_factory()
    if env_factory is not None:
        return env_factory
    return _default_lock


def _namespaced_channel(name: str) -> str:
    """Return the collision-resistant, namespaced channel for ``name``."""
    return f"crewai:{md5(name.encode(), usedforsecurity=False).hexdigest()}"


@contextmanager
def _default_lock(name: str, *, timeout: float = _DEFAULT_TIMEOUT) -> Iterator[None]:
    """The built-in backend: Redis when available, else a temp-dir file lock."""
    channel = _namespaced_channel(name)

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


def set_lock_backend(backend: LockBackend | None) -> None:
    """Override the locking backend used by :func:`lock`.

    Args:
        backend: A callable matching the :class:`LockBackend` protocol, i.e.
            ``backend(name, *, timeout) -> contextmanager``. Pass ``None`` to
            clear the override, falling back to the ``CREWAI_LOCK_FACTORY``
            env path if set, otherwise the built-in Redis/file default.
    """
    global _backend
    _backend = backend


def get_lock_backend() -> LockBackend:
    """Return the currently active locking backend.

    Honours the override > ``CREWAI_LOCK_FACTORY`` env > built-in default
    precedence.
    """
    return _active_backend()


@contextmanager
def lock(name: str, *, timeout: float = _DEFAULT_TIMEOUT) -> Iterator[None]:
    """Acquire a named lock, yielding while it is held.

    Delegates to the active backend, resolved as override >
    ``CREWAI_LOCK_FACTORY`` env > built-in Redis/file selection.

    Args:
        name: A human-readable lock name (e.g. ``"chromadb_init"``). The
              built-in default namespaces it to avoid collisions; custom
              backends receive it verbatim.
        timeout: Maximum seconds to wait for the lock before raising.
    """
    with _active_backend()(name, timeout=timeout):
        yield
