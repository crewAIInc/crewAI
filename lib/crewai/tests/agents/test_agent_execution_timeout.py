"""Regression tests for Agent.max_execution_time wall-clock enforcement."""

from __future__ import annotations

import threading
import time

import pytest

from crewai import Agent, Task


def test_timeout_is_raised_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout error must not wait for the runaway worker to finish."""
    release = threading.Event()

    def slow_execution(self: Agent, **kwargs: object) -> str:
        release.wait(2)
        return "done"

    monkeypatch.setattr(Agent, "_execute_without_timeout", slow_execution)
    agent = Agent(role="Researcher", goal="Test", backstory="Test")
    task = Task(description="slow task", expected_output="done")

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        agent._execute_with_timeout("prompt", task, 0.05)
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 1.5
