"""Regression tests for Agent.max_execution_time wall-clock enforcement."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from crewai import Agent, Task
from crewai.agents.agent_executor import AgentExecutor


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


def test_kickoff_is_raised_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standalone kickoff must stop at max_execution_time, not when the work ends."""
    release = threading.Event()

    def slow_invoke(self: AgentExecutor, inputs: dict[str, str]) -> dict[str, str]:
        release.wait(3)
        return {"output": "done"}

    monkeypatch.setattr(AgentExecutor, "invoke", slow_invoke)
    agent = Agent(role="Researcher", goal="Test", backstory="Test", max_execution_time=1)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        agent.kickoff("slow question")
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 2


def test_kickoff_async_is_raised_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """kickoff_async must apply the same deadline as the sync path."""

    async def slow_invoke_async(self: AgentExecutor, inputs: dict[str, str]) -> dict[str, str]:
        await asyncio.sleep(3)
        return {"output": "done"}

    monkeypatch.setattr(AgentExecutor, "invoke_async", slow_invoke_async)
    agent = Agent(role="Researcher", goal="Test", backstory="Test", max_execution_time=1)

    async def run() -> float:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await agent.kickoff_async("slow question")
        return time.monotonic() - started

    assert asyncio.run(run()) < 2


def test_kickoff_rejects_non_positive_max_execution_time() -> None:
    """A non-positive limit is a configuration error, not a no-op."""
    agent = Agent(role="Researcher", goal="Test", backstory="Test", max_execution_time=0)

    with pytest.raises(ValueError):
        agent.kickoff("question")
