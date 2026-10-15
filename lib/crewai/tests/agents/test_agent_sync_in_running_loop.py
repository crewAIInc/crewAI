"""Regression tests for crewAIInc/crewAI#8209.

Sync ``crew.kickoff()`` must keep working when called from inside a running
event loop (e.g. a Jupyter/Colab cell) — the contract that held for
1.14.0-1.14.4. Since 1.14.5, ``AgentExecutor.invoke()`` returns a coroutine in
that situation (magic auto-async for Flows) and
``Agent._execute_without_timeout`` raised ``RuntimeError`` for it.

These tests use a no-API-key stub LLM whose ``call()`` immediately returns a
final answer, mirroring the reporter's repro.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM


class StubLLM(BaseLLM):
    """No-API-key stub LLM: every call returns an immediate final answer."""

    def __init__(self) -> None:
        super().__init__(model="stub-model")

    def call(self, messages: Any, **kwargs: Any) -> str:
        return "Final Answer: ok"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


def _make_crew() -> Crew:
    agent = Agent(role="Researcher", goal="Answer", backstory="Stub", llm=StubLLM())
    task = Task(description="Say ok", expected_output="ok", agent=agent)
    return Crew(agents=[agent], tasks=[task])


def test_sync_kickoff_outside_event_loop_still_works():
    """Baseline: plain sync kickoff with no loop running is unchanged."""
    result = _make_crew().kickoff()
    assert result.raw == "ok"


def test_sync_kickoff_inside_running_event_loop():
    """Issue #8209: sync kickoff() from inside a running loop must not raise.

    Reproduces the Jupyter/Colab scenario: a loop is running on the caller
    thread, so AgentExecutor.invoke() returns a coroutine. The sync contract
    requires it to be driven to completion instead of raising RuntimeError.
    """

    async def run() -> str:
        result = _make_crew().kickoff()
        return result.raw

    assert asyncio.run(run()) == "ok"


def test_kickoff_async_still_takes_the_awaited_path():
    """The async entry point keeps working unchanged from a running loop."""

    async def run() -> str:
        result = await _make_crew().kickoff_async()
        return result.raw

    assert asyncio.run(run()) == "ok"


def test_concurrent_invocation_guard_still_fires():
    """An executor already running its async path must not be re-entered.

    When the same executor instance is mid-flight (its async path executing),
    a sync kickoff from the loop must surface the executor's own
    concurrency guard rather than deadlocking or silently re-entering.
    """
    crew = _make_crew()
    crew.kickoff()  # initializes agent.agent_executor
    agent = crew.agents[0]
    executor = agent.agent_executor
    assert executor is not None

    async def run() -> None:
        with pytest.raises(RuntimeError, match="Executor is already running"):
            with executor._execution_lock:
                executor._is_executing = True
            try:
                task = crew.tasks[0]
                agent._execute_without_timeout("Say ok", task)
            finally:
                with executor._execution_lock:
                    executor._is_executing = False

    asyncio.run(run())
