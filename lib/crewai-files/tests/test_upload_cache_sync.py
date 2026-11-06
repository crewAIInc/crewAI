"""Tests for UploadCache's sync wrappers across event-loop contexts."""

from __future__ import annotations

import asyncio
import time

from crewai_files.cache.upload_cache import UploadCache


FILE_HASH = "a" * 64


def test_sync_wrappers_round_trip_outside_event_loop() -> None:
    cache = UploadCache()

    cache.set_by_hash(FILE_HASH, "image/png", "openai", "file-1")

    cached = cache.get_by_hash(FILE_HASH, "openai")
    assert cached is not None
    assert cached.file_id == "file-1"


def test_sync_wrappers_from_running_loop_return_promptly() -> None:
    """Sync wrappers called on the loop's own thread must not deadlock.

    ``_run_sync`` used to schedule the coroutine on the running loop and
    block the loop thread on the result — unreachable progress, resolved
    only by the 30s timeout. A Jupyter kernel or any async app calling
    the sync API hits exactly this path.
    """

    async def scenario() -> None:
        cache = UploadCache()
        cache.set_by_hash(FILE_HASH, "image/png", "openai", "file-2")
        cached = cache.get_by_hash(FILE_HASH, "openai")
        assert cached is not None
        assert cached.file_id == "file-2"

    start = time.monotonic()
    asyncio.run(scenario())
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"sync wrappers took {elapsed:.1f}s inside a running loop"


def test_sync_write_from_loop_is_visible_to_async_read() -> None:
    """Entries written via the sync path inside a loop read back async."""

    async def scenario() -> None:
        cache = UploadCache()
        cache.set_by_hash(FILE_HASH, "image/png", "openai", "file-3")
        cached = await cache.aget_by_hash(FILE_HASH, "openai")
        assert cached is not None
        assert cached.file_id == "file-3"

    asyncio.run(scenario())
