from __future__ import annotations

import asyncio
from collections.abc import Coroutine
import concurrent.futures
import contextvars
from typing import Any, TypeVar


T = TypeVar("T")


def run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine synchronously, handling an already-running event loop.

    When called from within a running event loop (for example a FastAPI
    request handler, a Jupyter cell or any ``async def``), ``asyncio.run``
    raises ``RuntimeError: asyncio.run() cannot be called from a running event
    loop``. In that case the coroutine is executed to completion in a
    dedicated worker thread so the calling loop is neither blocked nor
    re-entered. Otherwise it falls back to ``asyncio.run``.
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
