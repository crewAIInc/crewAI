"""Run thread-based work inline, for runtimes that cannot start threads.

WebAssembly builds of CPython have no threads. crewai uses thread pools to run
blocking work (event handlers, tool calls, timeouts) off the calling thread;
running that work on the calling thread instead keeps the same results, only
without parallelism.
"""

from __future__ import annotations

from collections.abc import Callable
import concurrent.futures
import concurrent.futures.thread
import threading
from typing import Any, TypeVar


T = TypeVar("T")


class InlineExecutor(concurrent.futures.Executor):
    """Executor that runs each submitted call immediately, on the caller."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._shutdown = False

    def submit(
        self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
    ) -> concurrent.futures.Future[T]:
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")
        future: concurrent.futures.Future[T] = concurrent.futures.Future()
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as e:
                future.set_exception(e)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._shutdown = True


def _start_inline(self: threading.Thread) -> None:
    """``Thread.start`` replacement: run the target to completion now."""
    self.run()


def _join_inline(self: threading.Thread, timeout: float | None = None) -> None:
    """``Thread.join`` replacement: an inline thread has already finished."""


def install_inline_threads() -> None:
    """Replace thread pools and thread starts with inline execution."""
    concurrent.futures.ThreadPoolExecutor = InlineExecutor  # type: ignore[misc,assignment]
    concurrent.futures.thread.ThreadPoolExecutor = InlineExecutor  # type: ignore[misc,assignment]
    threading.Thread.start = _start_inline  # type: ignore[method-assign]
    threading.Thread.join = _join_inline  # type: ignore[method-assign]
