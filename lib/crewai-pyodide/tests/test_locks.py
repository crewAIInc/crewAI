from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import threading

from crewai_core import lock_store
import pytest

from crewai_pyodide._locks import in_process_lock, install_in_process_locks


@contextmanager
def held_in_another_thread(name: str) -> Iterator[None]:
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with in_process_lock(name, timeout=1):
            held.set()
            release.wait()

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    try:
        yield
    finally:
        release.set()
        holder.join()


def test_in_process_lock_is_reentrant() -> None:
    with in_process_lock("reentrant", timeout=1):
        with in_process_lock("reentrant", timeout=1):
            pass


def test_in_process_lock_times_out_when_held_elsewhere() -> None:
    with held_in_another_thread("contended"):
        with pytest.raises(TimeoutError):
            with in_process_lock("contended", timeout=0.05):
                pass


def test_lock_store_uses_in_process_locks_once_installed() -> None:
    install_in_process_locks()
    try:
        with held_in_another_thread("shared"):
            with pytest.raises(TimeoutError):
                with lock_store.lock("shared", timeout=0.05):
                    pass
    finally:
        lock_store.set_lock_backend(None)
