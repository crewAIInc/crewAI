"""Shared background event loop for Valkey GLIDE clients.

A GlideClient (and the asyncio.Lock guarding its creation) is bound to the
event loop it was created on. The Valkey cache and storage are called from many
loops: ``asyncio.run()`` per sync call, the caller's own loop, several threads
at once. Creating a client per loop races on rebind, and valkey-glide is
unreliable when many clients are created and closed on short-lived loops in
parallel threads (commands stop getting responses).

Instead, all GLIDE work runs on one long-lived loop in a daemon thread. Each
cache or storage instance keeps a single client that is only ever touched on
that loop, and callers on other loops await the result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
import functools
import threading
from typing import Any, ParamSpec, TypeVar


P = ParamSpec("P")
T = TypeVar("T")

_io_loop: asyncio.AbstractEventLoop | None = None
_io_loop_lock = threading.Lock()


def get_io_loop() -> asyncio.AbstractEventLoop:
    """Return the shared Valkey I/O loop, starting its thread on first use."""
    global _io_loop
    loop = _io_loop
    if loop is not None and loop.is_running():
        return loop
    with _io_loop_lock:
        if _io_loop is None or not _io_loop.is_running():
            new_loop = asyncio.new_event_loop()
            started = threading.Event()
            new_loop.call_soon(started.set)
            threading.Thread(
                target=new_loop.run_forever, name="valkey-io", daemon=True
            ).start()
            started.wait()
            _io_loop = new_loop
        return _io_loop


async def run_on_io_loop(coro: Coroutine[Any, Any, T]) -> T:
    """Await ``coro`` on the I/O loop from any event loop.

    Runs it directly if already on the I/O loop. Otherwise it is scheduled
    there; contextvars carry over and cancelling the caller cancels it.
    """
    loop = get_io_loop()
    if asyncio.get_running_loop() is loop:
        return await coro
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))


def run_on_io_loop_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on the I/O loop and block until it finishes.

    For synchronous callers. Must not be called from the I/O loop thread.
    """
    return asyncio.run_coroutine_threadsafe(coro, get_io_loop()).result()


def on_io_loop(
    fn: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Decorate an async method so its body always runs on the I/O loop."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return await run_on_io_loop(fn(*args, **kwargs))

    return wrapper
