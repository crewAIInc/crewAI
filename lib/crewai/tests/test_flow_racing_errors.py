"""Regression tests: racing listener batches must not swallow failures.

A failing non-racing listener in a batch that also contains racing listeners
used to be silently discarded (gather with ``return_exceptions=True`` and no
re-raise), so the flow "succeeded" even though the same failure crashes a
plain listener batch. Cancelled racers were also never awaited.
"""

import asyncio

import pytest

from crewai.flow import Flow, listen, or_, start


@pytest.mark.asyncio
async def test_failing_non_racing_listener_fails_flow_in_racing_batch():
    """The racing batch must fail the flow like a plain listener batch does."""
    fired: list[str] = []

    class RacingFlow(Flow):
        @start()
        def kick(self):
            return "k"

        @listen(kick)
        async def slow_a(self):
            await asyncio.sleep(1.0)
            return "a"

        @listen(kick)
        async def fast_b(self):
            return "b"

        @listen(or_(slow_a, fast_b))
        def handler(self):
            fired.append("handler")

        @listen(kick)
        def failing(self):
            raise RuntimeError("listener exploded")

    with pytest.raises(RuntimeError, match="listener exploded"):
        await RacingFlow().kickoff_async()


@pytest.mark.asyncio
async def test_racing_winner_and_cancellation_preserved_without_failures():
    """Guard: first-wins firing and racer cancellation are unchanged."""
    fired: list[str] = []
    a_cancelled = asyncio.Event()

    class RacingFlow(Flow):
        @start()
        def kick(self):
            return "k"

        @listen(kick)
        async def slow_a(self):
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                a_cancelled.set()
                raise
            return "a"

        @listen(kick)
        async def fast_b(self):
            return "b"

        @listen(or_(slow_a, fast_b))
        def handler(self):
            fired.append("handler")

    await RacingFlow().kickoff_async()

    assert fired == ["handler"]
    assert a_cancelled.is_set(), "losing racer should have been cancelled"
    leftovers = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert not leftovers, f"leaked background tasks: {leftovers}"
