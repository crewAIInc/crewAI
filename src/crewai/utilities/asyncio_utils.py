"""Utilities for handling asyncio operations safely across different contexts."""

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_coroutine_sync(coro: Coroutine) -> Any:
    """
    Run a coroutine synchronously, handling both cases where an event loop
    is already running and where it's not.

    This is useful when you need to run async code from sync code, but you're
    not sure if you're already in an async context (e.g., when using asyncio.to_thread).

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Raises:
        Any exception raised by the coroutine
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import threading

        result = None
        exception = None

        def run_in_new_loop():
            nonlocal result, exception
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            except Exception as e:
                exception = e

        thread = threading.Thread(target=run_in_new_loop)
        thread.start()
        thread.join()

        if exception:
            raise exception
        return result
