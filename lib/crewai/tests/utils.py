"""Test utilities for CrewAI tests."""

import asyncio
from concurrent.futures import ThreadPoolExecutor


def wait_for_event_handlers(timeout: float = 5.0) -> None:
    """Wait for all pending event handlers to complete.

    This helper ensures all sync and async handlers finish processing before
    proceeding. Useful in tests to make assertions deterministic.

    Args:
        timeout: Maximum time to wait in seconds.
    """
    from crewai.events.event_bus import crewai_event_bus

    loop = getattr(crewai_event_bus, "_loop", None)

    if loop and not loop.is_closed():

        async def _wait_for_async_tasks() -> None:
            tasks = {
                t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()
            }
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        future = asyncio.run_coroutine_threadsafe(_wait_for_async_tasks(), loop)
        try:
            future.result(timeout=timeout)
        except Exception:  # noqa: S110
            pass

    crewai_event_bus._sync_executor.shutdown(wait=True)
    crewai_event_bus._sync_executor = ThreadPoolExecutor(
        max_workers=10,
        thread_name_prefix="CrewAISyncHandler",
    )
