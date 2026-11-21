"""Regression tests for Agent.max_execution_time wall-clock enforcement."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

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


def test_kickoff_raises_timeout_error_at_max_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    worker_done = threading.Event()

    def slow_invoke(self: AgentExecutor, inputs: dict[str, Any]) -> dict[str, str]:
        try:
            release.wait(4)
            return {"output": "done"}
        finally:
            worker_done.set()

    monkeypatch.setattr(AgentExecutor, "invoke", slow_invoke)
    agent = Agent(
        role="Researcher",
        goal="Test",
        backstory="Test",
        max_execution_time=1,
    )

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
            agent.kickoff("slow question")
        elapsed = time.monotonic() - started
    finally:
        release.set()
        assert worker_done.wait(2)

    assert elapsed < 2.5


def test_kickoff_async_raises_timeout_error_at_max_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_invoke_async(
        self: AgentExecutor, inputs: dict[str, Any]
    ) -> dict[str, str]:
        await asyncio.sleep(4)
        return {"output": "done"}

    monkeypatch.setattr(AgentExecutor, "invoke_async", slow_invoke_async)
    agent = Agent(
        role="Researcher",
        goal="Test",
        backstory="Test",
        max_execution_time=1,
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
        asyncio.run(agent.kickoff_async("slow question"))
    elapsed = time.monotonic() - started

    assert elapsed < 2.5


@pytest.mark.parametrize("is_async", [False, True])
def test_kickoff_rejects_non_positive_max_execution_time(
    monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    async def fast_invoke_async(
        self: AgentExecutor, inputs: dict[str, Any]
    ) -> dict[str, str]:
        return {"output": "done"}

    monkeypatch.setattr(AgentExecutor, "invoke", lambda self, inputs: {"output": "done"})
    monkeypatch.setattr(AgentExecutor, "invoke_async", fast_invoke_async)
    agent = Agent(
        role="Researcher",
        goal="Test",
        backstory="Test",
        max_execution_time=0,
    )

    with pytest.raises(ValueError, match="positive integer greater than zero"):
        if is_async:
            asyncio.run(agent.kickoff_async("hello"))
        else:
            agent.kickoff("hello")


def test_kickoff_returns_output_within_max_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fast_invoke(self: AgentExecutor, inputs: dict[str, Any]) -> dict[str, str]:
        return {"output": "done"}

    monkeypatch.setattr(AgentExecutor, "invoke", fast_invoke)
    agent = Agent(
        role="Researcher",
        goal="Test",
        backstory="Test",
        max_execution_time=5,
    )

    result = agent.kickoff("quick question")
    assert result.raw == "done"
