"""Utilities for running async code from sync contexts."""

import asyncio
import concurrent.futures
import contextvars
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine synchronously, handling an already-running event loop.

    ``asyncio.run()`` raises ``RuntimeError`` when called from within a running
    event loop (e.g. inside a FastAPI handler, Jupyter cell, or any ``async
    def``).  This helper detects that case and offloads the coroutine to a
    dedicated thread with its own loop, preserving the calling context.
    """
    try:
        asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        has_running_loop = False

    if has_running_loop:
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(ctx.run, asyncio.run, coro).result()
    return asyncio.run(coro)
