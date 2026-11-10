"""_execute_with_timeout must report TimeoutError at the deadline, not after
the worker finishes (#9323)."""

import threading
import time

import pytest

from crewai import Agent


@pytest.fixture()
def agent():
    return Agent(role="r", goal="g", backstory="b")


def _patch_execute(agent, release, started):
    def slow(self, **kwargs):
        started.set()
        release.wait(timeout=5)
        return "done"

    agent._execute_without_timeout = slow.__get__(agent)


def test_timeout_raises_at_deadline_not_after_worker_finishes(agent):
    """A task sleeping longer than max_execution_time must raise TimeoutError
    on time. shutdown(wait=True) used to block the caller until the worker
    returned (~3s for a 1s timeout)."""
    release = threading.Event()
    started = threading.Event()
    _patch_execute(agent, release, started)

    try:
        begin = time.monotonic()
        with pytest.raises(TimeoutError):
            agent._execute_with_timeout("prompt", type("T", (), {"description": "task"})(), 1)
        elapsed = time.monotonic() - begin
    finally:
        release.set()

    assert started.wait(timeout=5)
    assert elapsed < 2.0, f"TimeoutError took {elapsed:.1f}s; shutdown(wait=True) is blocking the caller"


def test_success_path_still_returns_result(agent):
    def fast(self, **kwargs):
        return "done"

    agent._execute_without_timeout = fast.__get__(agent)
    assert agent._execute_with_timeout("prompt", type("T", (), {"description": "task"})(), 5) == "done"
