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
import threading
from typing import TypeVar


# The pristine C-level connect, kept out of reach of guards that replace the
# class attribute on socket.socket.
_PRISTINE_CONNECT = _socket.socket.connect
_PRISTINE_CONNECT_EX = _socket.socket.connect_ex

# Serializes context entry/exit so overlapping uses cannot permanently replace
# a guard with the pristine connect.
_GUARD_SWAP_LOCK = threading.RLock()

T = TypeVar("T")


@contextmanager
def unpatched_socket_connect() -> Generator[None, None, None]:
    """Temporarily restore the built-in ``socket.socket.connect``.

    Scope: only the asyncio loop construction performed by the caller. The
    guard is put back exactly as it was afterwards, so real network calls
    stay blocked. On POSIX this is a no-op in practice: the loop self-pipe is
    created with ``os.socketpair()`` and never calls ``connect()``.
    """
    with _GUARD_SWAP_LOCK:
        previous_connect = socket.socket.connect
        previous_connect_ex = socket.socket.connect_ex
        socket.socket.connect = _PRISTINE_CONNECT
        socket.socket.connect_ex = _PRISTINE_CONNECT_EX
        try:
            yield
        finally:
            socket.socket.connect = previous_connect
            socket.socket.connect_ex = previous_connect_ex


def run_in_new_loop(coro: Coroutine[None, None, T]) -> T:
    """Run *coro* in a fresh event loop, exempting only its construction.

    Equivalent to ``asyncio.run(coro)`` except that the socket guard is
    lifted while the loop object (and its Windows self-pipe) is built, and
    reinstated before the coroutine runs.
    """
    with unpatched_socket_connect():
        loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
