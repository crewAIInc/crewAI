"""Regression tests for Agent.max_execution_time wall-clock enforcement."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from crewai import Agent, Task
from crewai.agent.core import LiteAgentOutput


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


def test_agent_kickoff_enforces_max_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent.kickoff must raise TimeoutError when max_execution_time is exceeded."""
    release = threading.Event()

    def slow_execute(self: Agent, *args: object, **kwargs: object) -> object:
        release.wait(2)
        return LiteAgentOutput(raw="done", agent_role="Researcher")

    monkeypatch.setattr(Agent, "_execute_and_build_output", slow_execute)
    agent = Agent(
        role="Researcher", goal="Test", backstory="Test", max_execution_time=1
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="execution timed out after 1 seconds"):
        agent.kickoff("hello")
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_agent_kickoff_async_enforces_max_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent.kickoff_async must raise TimeoutError when max_execution_time is exceeded."""

    async def slow_execute_async(
        self: Agent, *args: object, **kwargs: object
    ) -> object:
        await asyncio.sleep(2)
        return LiteAgentOutput(raw="done", agent_role="Researcher")

    monkeypatch.setattr(Agent, "_execute_and_build_output_async", slow_execute_async)
    agent = Agent(
        role="Researcher", goal="Test", backstory="Test", max_execution_time=1
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="execution timed out after 1 seconds"):
        await agent.kickoff_async("hello")
    elapsed = time.monotonic() - started

    assert elapsed < 1.5
