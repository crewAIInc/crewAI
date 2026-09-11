"""Tests for crew-level token usage aggregation.

Every LLM call is credited to the agent that made it at the moment the
provider reports usage, and ``crew.usage_metrics`` sums those per-agent
totals for the run. These tests drive real kickoffs through an offline LLM
that reports usage the way providers do, so they exercise the same
attribution path as production.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import PrivateAttr

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM


TOKENS_PER_CALL = 100
FINAL_ANSWER = "Thought: I know the answer.\nFinal Answer: done"


class _UsageLLM(BaseLLM):
    """Offline LLM that reports ``TOKENS_PER_CALL`` tokens on every call.

    ``responses`` are returned first, in order; after that every call gives
    a final answer.
    """

    _delay: float = PrivateAttr(default=0.0)
    _responses: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, delay: float = 0.0, responses: list[str] | None = None) -> None:
        super().__init__(model="fake-usage-model")
        self._delay = delay
        self._responses = list(responses or [])

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
        if self._delay:
            time.sleep(self._delay)
        self._track_token_usage_internal(
            {"prompt_tokens": TOKENS_PER_CALL, "completion_tokens": 0}
        )
        return self._responses.pop(0) if self._responses else FINAL_ANSWER

    async def acall(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        return self.call(messages)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096

    @property
    def spent(self) -> int:
        """Tokens this instance has actually consumed."""
        return self.get_token_usage_summary().total_tokens


def _agent(role: str, llm: BaseLLM) -> Agent:
    return Agent(role=role, goal="goal", backstory="backstory", llm=llm)


def _task(agent: Agent, name: str = "task", async_execution: bool = False) -> Task:
    return Task(
        description=name,
        expected_output="out",
        agent=agent,
        async_execution=async_execution,
    )


def test_agents_sharing_one_llm_are_counted_once() -> None:
    """Three agents on one instance report what it consumed, not 3x it."""
    llm = _UsageLLM()
    agents = [_agent(f"Role {i}", llm) for i in range(3)]
    crew = Crew(agents=agents, tasks=[_task(a, f"task {i}") for i, a in enumerate(agents)])

    crew.kickoff()

    assert llm.spent > 0
    assert crew.usage_metrics.total_tokens == llm.spent


def test_each_agent_is_credited_with_its_own_calls() -> None:
    """A shared instance's usage is split between the agents that made the calls."""
    llm = _UsageLLM()
    agents = [_agent(f"Role {i}", llm) for i in range(3)]
    crew = Crew(agents=agents, tasks=[_task(a, f"task {i}") for i, a in enumerate(agents)])

    crew.kickoff()

    shares = [agent._usage_metrics.total_tokens for agent in agents]
    assert all(share > 0 for share in shares)
    assert sum(shares) == llm.spent


def test_distinct_llm_instances_are_summed() -> None:
    """Separate instances still add up, even for the same model."""
    first, second = _UsageLLM(), _UsageLLM()
    a, b = _agent("A", first), _agent("B", second)
    crew = Crew(agents=[a, b], tasks=[_task(a, "a"), _task(b, "b")])

    crew.kickoff()

    assert crew.usage_metrics.total_tokens == first.spent + second.spent


def test_an_agent_on_two_tasks_is_counted_once_per_call() -> None:
    """An agent assigned to two tasks is credited with both, and only once."""
    llm = _UsageLLM()
    agent = _agent("Solo", llm)
    crew = Crew(agents=[agent], tasks=[_task(agent, "first"), _task(agent, "second")])

    crew.kickoff()

    assert llm.spent >= 2 * TOKENS_PER_CALL
    assert agent._usage_metrics.total_tokens == llm.spent
    assert crew.usage_metrics.total_tokens == llm.spent


def test_overlapping_async_tasks_on_a_shared_llm_are_not_double_counted() -> None:
    """Async tasks running at once on one LLM instance each count their own calls.

    A snapshot taken around each task would see the other task's calls inside
    its window and count them twice.
    """
    llm = _UsageLLM(delay=0.05)
    a, b = _agent("A", llm), _agent("B", llm)
    crew = Crew(
        agents=[a, b],
        tasks=[
            _task(a, "first", async_execution=True),
            _task(b, "second", async_execution=True),
            _task(a, "third"),
        ],
    )

    crew.kickoff()

    assert a._usage_metrics.total_tokens + b._usage_metrics.total_tokens == llm.spent
    assert crew.usage_metrics.total_tokens == llm.spent


def test_a_later_run_excludes_the_previous_one() -> None:
    """A second kickoff reports only its own usage, request counts included."""
    llm = _UsageLLM()
    agent = _agent("Solo", llm)
    crew = Crew(agents=[agent], tasks=[_task(agent)])

    crew.kickoff()
    assert crew.usage_metrics == llm.get_token_usage_summary()

    before = llm.get_token_usage_summary()
    crew.kickoff()

    assert crew.usage_metrics == llm.get_token_usage_summary().delta_since(before)
    assert crew.usage_metrics.successful_requests >= 1


def test_a_manager_llm_shared_by_two_crews_reports_each_run() -> None:
    """A manager created during kickoff is credited from its first call."""
    manager_llm = _UsageLLM()

    def hierarchical_crew() -> tuple[Crew, _UsageLLM]:
        worker_llm = _UsageLLM()
        worker = _agent("Worker", worker_llm)
        crew = Crew(
            agents=[worker],
            tasks=[_task(worker)],
            process=Process.hierarchical,
            manager_llm=manager_llm,
        )
        return crew, worker_llm

    first, first_worker = hierarchical_crew()
    first.kickoff()
    after_first = manager_llm.spent
    assert after_first > 0
    assert first.usage_metrics.total_tokens == after_first + first_worker.spent

    second, second_worker = hierarchical_crew()
    second.kickoff()
    assert manager_llm.spent > after_first
    assert second.usage_metrics.total_tokens == (
        manager_llm.spent - after_first
    ) + second_worker.spent


def test_delegated_work_is_credited_to_the_coworker() -> None:
    """A coworker's calls count toward the crew when the manager delegates."""
    delegate = (
        "Thought: The worker should handle this.\n"
        "Action: Delegate work to coworker\n"
        'Action Input: {"task": "say done", "context": "none", "coworker": "Worker"}'
    )
    manager_llm = _UsageLLM(responses=[delegate])
    worker_llm = _UsageLLM()
    worker = _agent("Worker", worker_llm)
    crew = Crew(
        agents=[worker],
        tasks=[_task(worker)],
        process=Process.hierarchical,
        manager_llm=manager_llm,
    )

    crew.kickoff()

    assert worker_llm.spent > 0
    assert worker._usage_metrics.total_tokens == worker_llm.spent
    assert crew.usage_metrics.total_tokens == manager_llm.spent + worker_llm.spent


def test_concurrent_kickoff_for_each_async_is_not_inflated() -> None:
    """Crew copies running at once each report only their own calls."""
    llm = _UsageLLM(delay=0.05)
    agent = _agent("Solo", llm)
    crew = Crew(agents=[agent], tasks=[_task(agent, "task {x}")])

    asyncio.run(crew.kickoff_for_each_async(inputs=[{"x": 1}, {"x": 2}, {"x": 3}]))

    assert llm.spent >= 3 * TOKENS_PER_CALL
    assert crew.usage_metrics.total_tokens == llm.spent
