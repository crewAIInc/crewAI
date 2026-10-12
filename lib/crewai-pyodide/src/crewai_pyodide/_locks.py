"""In-process locks for ``crewai_core.lock_store``.

portalocker's file locks need ``fcntl``, which WebAssembly builds of CPython
lack. Pyodide only ever runs one process, so a re-entrant lock per name gives
the same guarantees.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
import threading


_locks: defaultdict[str, threading.RLock] = defaultdict(threading.RLock)


@contextmanager
def in_process_lock(name: str, *, timeout: float) -> Iterator[None]:
    """Hold the lock named ``name``; raise TimeoutError after ``timeout`` seconds."""
    lock = _locks[name]
    if not lock.acquire(timeout=timeout):
        raise TimeoutError(f"Failed to acquire lock '{name}' (timeout={timeout}s)")
    try:
        yield
    finally:
        lock.release()


def install_in_process_locks() -> None:
    """Make ``crewai_core.lock_store.lock`` use :func:`in_process_lock`."""
    from crewai_core.lock_store import set_lock_backend

    set_lock_backend(in_process_lock)
