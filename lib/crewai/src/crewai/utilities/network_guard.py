"""Helpers for building asyncio event loops under socket-blocking test guards.

Windows emulates ``socket.socketpair()`` with a real ``connect()`` to 127.0.0.1,
and asyncio builds every event loop's self-pipe that way. Socket-blocking test
guards (pytest-recording's ``--block-network``, pytest-socket's
``--disable-socket``) replace ``socket.socket.connect`` and reject that
internal, non-network self-pipe, so loop construction dies with
"Network is disabled" on Windows. Linux builds the self-pipe via
``os.socketpair()`` and never calls ``connect()``, which is why CI never sees
this.
"""

from __future__ import annotations

import _socket
import asyncio
from collections.abc import Coroutine, Generator
from contextlib import contextmanager
import socket
import sys
import threading
from typing import TypeVar


# The pristine C-level connect, kept out of reach of guards that replace the
# class attribute on socket.socket.
_PRISTINE_CONNECT = _socket.socket.connect
_PRISTINE_CONNECT_EX = _socket.socket.connect_ex

# Serializes context entry/exit. The thread holding this lock while a context
# is open is the only thread exempted from an installed guard.
_GUARD_SWAP_LOCK = threading.RLock()
_EXEMPT_THREAD: threading.Thread | None = None

T = TypeVar("T")


@contextmanager
def unpatched_socket_connect() -> Generator[None, None, None]:
    """Exempt the calling thread from any installed socket guard.

    Scope: only the asyncio loop construction performed by the caller, and
    only sockets connected from the calling thread — every other thread keeps
    hitting the guard, so the exemption never opens a process-wide window.
    The guard is put back exactly as it was afterwards, so real network calls
    stay blocked. On POSIX this is a no-op in practice: the loop self-pipe is
    created with ``os.socketpair()`` and never calls ``connect()``.
    """
    global _EXEMPT_THREAD
    with _GUARD_SWAP_LOCK:
        previous_connect = socket.socket.connect
        previous_connect_ex = socket.socket.connect_ex
        nested = _EXEMPT_THREAD is threading.current_thread()

        def exempting_connect(sock, address, *args, **kwargs):
            if threading.current_thread() is _EXEMPT_THREAD:
                return _PRISTINE_CONNECT(sock, address, *args, **kwargs)
            return previous_connect(sock, address, *args, **kwargs)

        def exempting_connect_ex(sock, address, *args, **kwargs):
            if threading.current_thread() is _EXEMPT_THREAD:
                return _PRISTINE_CONNECT_EX(sock, address, *args, **kwargs)
            return previous_connect_ex(sock, address, *args, **kwargs)

        if not nested:
            _EXEMPT_THREAD = threading.current_thread()
        socket.socket.connect = exempting_connect
        socket.socket.connect_ex = exempting_connect_ex
        try:
            yield
        finally:
            socket.socket.connect = previous_connect
            socket.socket.connect_ex = previous_connect_ex
            if not nested:
                _EXEMPT_THREAD = None


def run_in_new_loop(coro: Coroutine[None, None, T]) -> T:
    """Run *coro* in a fresh event loop, exempting only its construction.

    Equivalent to ``asyncio.run(coro)`` except that the socket guard is
    lifted (for the calling thread) while the loop object (and its Windows
    self-pipe) is built, and reinstated before the coroutine runs. Shutdown
    mirrors ``asyncio.run``: pending tasks are cancelled, async generators
    and the default executor are shut down, then the loop is closed.
    """
    with unpatched_socket_connect():
        loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                for task in pending:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        loop.call_exception_handler(
                            {
                                "message": (
                                    "unhandled exception during event loop shutdown"
                                ),
                                "exception": exc,
                                "task": task,
                            }
                        )
            loop.run_until_complete(loop.shutdown_asyncgens())
            if sys.version_info >= (3, 12):
                # asyncio.run bounds the default-executor wait on these
                # versions; keep the same guarantee here.
                loop.run_until_complete(loop.shutdown_default_executor(timeout=5 * 60))
            else:
                loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
