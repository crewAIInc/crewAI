"""Tests for concurrent async tasks assigned to one agent.

An ``Agent`` holds a single ``agent_executor`` whose state is reset at the
start of every ``invoke``. Two ``async_execution=True`` tasks on the same agent
run concurrently, so sharing one executor between them corrupted that state and
the executor's own guard rejected the second invocation with
``RuntimeError: Executor is already running``.

Each async task now gets an executor bound to the thread running it. A stub LLM
keeps these offline; no provider is required.
"""

from __future__ import annotations

import threading
from typing import Any

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM


class _StubLLM(BaseLLM):
    """Offline LLM that answers immediately and records concurrency."""

    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            return "Thought: I know the answer.\nFinal Answer: done"
        finally:
            with self._lock:
                self.concurrent -= 1

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096


def _agent(llm: _StubLLM, role: str = "Worker") -> Agent:
    return Agent(role=role, goal="Answer briefly.", backstory="Brief.", llm=llm)


def test_two_async_tasks_on_one_agent_complete() -> None:
    """Two concurrent async tasks may share an agent without colliding."""
    llm = _StubLLM()
    agent = _agent(llm)
    tasks = [
        Task(
            description=f"say {word}",
            expected_output="one word",
            agent=agent,
            async_execution=True,
        )
        for word in ("alpha", "beta")
    ]
    tasks.append(
        Task(description="say done", expected_output="one word", agent=agent)
    )

    result = Crew(agents=[agent], tasks=tasks).kickoff()

    assert len(result.tasks_output) == 3
    assert all(output.raw for output in result.tasks_output)


def test_three_async_tasks_on_one_agent_complete() -> None:
    """The limit is not two: any number of async tasks may share an agent."""
    llm = _StubLLM()
    agent = _agent(llm)
    tasks = [
        Task(
            description=f"say w{index}",
            expected_output="one word",
            agent=agent,
            async_execution=True,
        )
        for index in range(3)
    ]
    tasks.append(
        Task(description="say done", expected_output="one word", agent=agent)
    )

    result = Crew(agents=[agent], tasks=tasks).kickoff()

    assert len(result.tasks_output) == 4


def test_async_tasks_do_not_reuse_one_executor() -> None:
    """Concurrent tasks on one agent must not share an executor instance."""
    llm = _StubLLM()
    agent = _agent(llm)
    seen: list[int] = []
    lock = threading.Lock()

    original = Agent._active_executor

    def record(self: Agent) -> Any:
        executor = original(self)
        with lock:
            seen.append(id(executor))
        return executor

    Agent._active_executor = record  # type: ignore[method-assign]
    try:
        tasks = [
            Task(
                description=f"say w{index}",
                expected_output="one word",
                agent=agent,
                async_execution=True,
            )
            for index in range(2)
        ]
        tasks.append(
            Task(description="say done", expected_output="one word", agent=agent)
        )
        Crew(agents=[agent], tasks=tasks).kickoff()
    finally:
        Agent._active_executor = original  # type: ignore[method-assign]

    # two async tasks plus one sync task, each with a distinct executor
    assert len(set(seen)) == len(seen) == 3


def test_sequential_tasks_still_reuse_the_agent_executor() -> None:
    """Non-async tasks keep reusing the agent's own executor."""
    llm = _StubLLM()
    agent = _agent(llm)
    tasks = [
        Task(description=f"say s{index}", expected_output="one word", agent=agent)
        for index in range(2)
    ]

    Crew(agents=[agent], tasks=tasks).kickoff()

    assert agent.agent_executor is not None
