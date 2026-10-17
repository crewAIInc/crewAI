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
import threading
from typing import Any

import pytest

from crewai import Agent, Crew, Task
from crewai.agent.core import _run_awaitable_on_worker_thread
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


def test_reentrant_sync_invoke_runs_on_dedicated_thread():
    """A sync kickoff issued from a pool worker must not submit to the pool.

    An executor step callback runs on a pool worker thread (inside that
    worker's event loop); if it calls crew.kickoff() synchronously, the
    nested invocation must run on a dedicated thread. Submitting back into
    the bounded pool and blocking would deadlock once every worker is
    waiting on a queued job. The routing is asserted via thread names so
    the test is deterministic (no timing involved).
    """
    seen: dict[str, str] = {}

    async def _inner() -> str:
        seen["thread"] = threading.current_thread().name
        return "inner-ok"

    async def _outer() -> str:
        # Runs on a pool worker; the nested call must take the dedicated path.
        return _run_awaitable_on_worker_thread(_inner())

    assert _run_awaitable_on_worker_thread(_outer()) == "inner-ok"
    assert seen["thread"].startswith("crewai-sync-invoke-reentrant")


def test_reentrant_sync_invoke_propagates_exceptions():
    """Exceptions from a re-entrant nested invocation reach the caller."""

    async def _boom() -> str:
        raise ValueError("nested failure")

    async def _outer() -> str:
        return _run_awaitable_on_worker_thread(_boom())

    with pytest.raises(ValueError, match="nested failure"):
        _run_awaitable_on_worker_thread(_outer())


def test_fire_and_forget_callback_tasks_are_drained_not_dropped():
    """Fire-and-forget tasks on the worker loop must complete, not be dropped.

    The worker loop is torn down as soon as the main coroutine finishes; a
    task scheduled via asyncio.create_task during the run (the pattern
    _invoke_step_callback uses for async callbacks) must be given a chance
    to finish instead of being cancelled by asyncio.run() — otherwise the
    final AgentFinish callback would be silently dropped.
    """
    completed: list[str] = []

    async def _callback() -> None:
        await asyncio.sleep(0.01)
        completed.append("callback")

    async def _main() -> str:
        asyncio.create_task(_callback())
        return "main-ok"

    assert _run_awaitable_on_worker_thread(_main()) == "main-ok"
    assert completed == ["callback"]


def test_drain_timeout_bounds_misbehaving_callbacks(monkeypatch):
    """A callback that never finishes cannot hang the sync call."""
    import time

    import crewai.agent.core as core_module

    monkeypatch.setattr(core_module, "_CALLBACK_DRAIN_TIMEOUT_S", 0.05)

    async def _stuck() -> None:
        await asyncio.sleep(30)

    async def _main() -> str:
        asyncio.create_task(_stuck())
        return "main-ok"

    start = time.monotonic()
    assert _run_awaitable_on_worker_thread(_main()) == "main-ok"
    assert time.monotonic() - start < 10
