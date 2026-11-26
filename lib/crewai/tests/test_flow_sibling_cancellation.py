"""Regression tests: a failing flow method must not abandon its in-flight
siblings.

Both the plain listener batch (``_execute_listeners``) and the parallel
start-method batch in ``kickoff_async`` used a bare ``asyncio.gather``, which
propagates the first exception and leaves the remaining coroutines running in
the background after the flow has already failed.
"""

import asyncio

import pytest

from crewai.flow import Flow, listen, start


@pytest.mark.asyncio
async def test_failing_listener_cancels_sibling_listeners():
    cancelled = asyncio.Event()
    finished = asyncio.Event()
    fired: list[str] = []

    class F(Flow):
        @start()
        def kick(self):
            return "k"

        @listen(kick)
        async def slow(self):
            fired.append("slow-started")
            try:
                await asyncio.sleep(2.0)
                finished.set()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        @listen(kick)
        async def failer(self):
            await asyncio.sleep(0.05)
            raise RuntimeError("listener exploded")

    with pytest.raises(RuntimeError, match="listener exploded"):
        await F().kickoff_async()

    assert cancelled.is_set()
    assert not finished.is_set()
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert pending == []


@pytest.mark.asyncio
async def test_failing_start_cancels_sibling_starts():
    cancelled = asyncio.Event()
    finished = asyncio.Event()
    fired: list[str] = []

    class F(Flow):
        @start()
        async def slow_start(self):
            fired.append("slow-start-started")
            try:
                await asyncio.sleep(2.0)
                finished.set()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        @start()
        async def fail_start(self):
            await asyncio.sleep(0.05)
            raise RuntimeError("start exploded")

    with pytest.raises(RuntimeError, match="start exploded"):
        await F().kickoff_async()

    assert cancelled.is_set()
    assert not finished.is_set()
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    assert pending == []


@pytest.mark.asyncio
async def test_successful_batches_run_all_methods():
    """Guard: without failures, parallel listeners and starts all complete."""
    fired: list[str] = []

    class F(Flow):
        @start()
        async def start_a(self):
            await asyncio.sleep(0.05)
            fired.append("start_a")

        @start()
        def start_b(self):
            fired.append("start_b")

        @listen(start_a)
        def listener_a(self):
            fired.append("listener_a")

        @listen(start_a)
        async def listener_b(self):
            await asyncio.sleep(0.05)
            fired.append("listener_b")

    await F().kickoff_async()

    assert set(fired) == {"start_a", "start_b", "listener_a", "listener_b"}
