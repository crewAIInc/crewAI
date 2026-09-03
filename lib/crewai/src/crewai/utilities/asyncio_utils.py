"""Helpers for bridging coroutines into synchronous call sites."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
import concurrent.futures
import contextvars
from typing import Any, TypeVar


T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion from synchronous code.

    ``asyncio.run`` raises ``RuntimeError`` when a loop is already running on
    the current thread, which happens whenever synchronous code is reached from
    an async caller -- for example a tool invoked from an ``async`` Flow method.
    In that case run the coroutine on a worker thread, which has no loop of its
    own, carrying the caller's context variables across.

    Args:
        coro: The coroutine to execute.

    Returns:
        Whatever the coroutine returns.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    ctx = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()
