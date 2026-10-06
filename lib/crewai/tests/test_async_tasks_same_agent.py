"""Tests for concurrent async tasks assigned to one agent.

An ``Agent`` holds a single ``agent_executor`` whose state is reset at the
start of every ``invoke``. Two ``async_execution=True`` tasks on the same agent
run concurrently, so sharing one executor between them corrupted that state and
the executor's own guard rejected the second invocation with
``RuntimeError: Executor is already running``.

Each async task now gets an executor bound to the context running it, on both
``kickoff`` and ``akickoff``. A stub LLM keeps these offline; no provider is
required.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import threading
from typing import Any

import pytest

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool


FINAL_ANSWER = "Thought: I know the answer.\nFinal Answer: done"


class _StubLLM(BaseLLM):
    """Offline LLM that answers immediately and records concurrency.

    With ``overlap`` set, the first ``overlap`` calls wait for one another
    before answering, so they are guaranteed to be in flight together. A call
    that never gets company times out, failing its task.
    """

    def __init__(self, overlap: int = 0) -> None:
        super().__init__(model="stub-model")
        self.concurrent = 0
        self.max_concurrent = 0
        self.prompts: list[str] = []
        self._calls = 0
        self._overlap = overlap
        self._barrier = threading.Barrier(overlap, timeout=5) if overlap else None
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
            self.prompts.append(_text(messages))
            barrier = self._barrier if self._calls < self._overlap else None
            self._calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if barrier is not None:
                barrier.wait()
            return FINAL_ANSWER
        finally:
            with self._lock:
                self.concurrent -= 1

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096


class _ProbeTool(BaseTool):
    name: str = "probe_tool"
    description: str = "Given only to the async task."

    def _run(self, **kwargs: Any) -> str:
        return "probe"


Runner = Callable[[Crew], Any]


def _kickoff(crew: Crew) -> Any:
    return crew.kickoff()


def _akickoff(crew: Crew) -> Any:
    return asyncio.run(crew.akickoff())


both_paths = pytest.mark.parametrize(
    "run", [_kickoff, _akickoff], ids=["kickoff", "akickoff"]
)


def _text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(message.get("content", "")) for message in messages)


def _agent(llm: _StubLLM, role: str = "Worker") -> Agent:
    return Agent(role=role, goal="Answer briefly.", backstory="Brief.", llm=llm)


def _async_tasks(agent: Agent, count: int) -> list[Task]:
    """``count`` async tasks, then the sync task that waits for them."""
    tasks = [
        Task(
            description=f"say w{index}",
            expected_output="one word",
            agent=agent,
            async_execution=True,
        )
        for index in range(count)
    ]
    tasks.append(Task(description="say done", expected_output="one word", agent=agent))
    return tasks


def _record_executors(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the id of the executor each task execution runs on."""
    seen: list[int] = []
    lock = threading.Lock()
    original = Agent._active_executor

    def record(self: Agent) -> Any:
        executor = original(self)
        with lock:
            seen.append(id(executor))
        return executor

    monkeypatch.setattr(Agent, "_active_executor", record)
    return seen


@both_paths
def test_two_async_tasks_on_one_agent_overlap_and_complete(run: Runner) -> None:
    """Two async tasks may share an agent and run at the same time."""
    llm = _StubLLM(overlap=2)
    agent = _agent(llm)

    result = run(Crew(agents=[agent], tasks=_async_tasks(agent, 2)))

    assert llm.max_concurrent == 2
    assert len(result.tasks_output) == 3
    assert all(output.raw for output in result.tasks_output)


@both_paths
def test_three_async_tasks_on_one_agent_overlap_and_complete(run: Runner) -> None:
    """The limit is not two: any number of async tasks may share an agent."""
    llm = _StubLLM(overlap=3)
    agent = _agent(llm)

    result = run(Crew(agents=[agent], tasks=_async_tasks(agent, 3)))

    assert llm.max_concurrent == 3
    assert len(result.tasks_output) == 4


@both_paths
def test_async_tasks_do_not_reuse_one_executor(
    run: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent tasks on one agent must not share an executor instance."""
    seen = _record_executors(monkeypatch)
    agent = _agent(_StubLLM())

    run(Crew(agents=[agent], tasks=_async_tasks(agent, 2)))

    # two async tasks plus one sync task, each with a distinct executor
    assert len(set(seen)) == len(seen) == 3


@both_paths
def test_async_task_runs_with_its_own_tools(run: Runner) -> None:
    """An async task's executor is built for that task, its tools included."""
    llm = _StubLLM()
    agent = _agent(llm)
    tasks = [
        Task(
            description="say w0",
            expected_output="one word",
            agent=agent,
            async_execution=True,
            tools=[_ProbeTool()],
        ),
        Task(description="say done", expected_output="one word", agent=agent),
    ]

    run(Crew(agents=[agent], tasks=tasks))

    async_prompts = [prompt for prompt in llm.prompts if "say w0" in prompt]
    assert async_prompts
    assert all("probe_tool" in prompt for prompt in async_prompts)


@both_paths
def test_sequential_tasks_still_reuse_the_agent_executor(
    run: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-async tasks keep reusing the agent's own executor."""
    seen = _record_executors(monkeypatch)
    agent = _agent(_StubLLM())
    tasks = [
        Task(description=f"say s{index}", expected_output="one word", agent=agent)
        for index in range(2)
    ]

    run(Crew(agents=[agent], tasks=tasks))

    assert len(seen) == 2
    assert set(seen) == {id(agent.agent_executor)}
