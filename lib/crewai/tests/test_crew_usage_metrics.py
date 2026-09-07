"""Tests for crew-level token usage aggregation.

``crew.usage_metrics`` reports what a kickoff consumed, measured as the growth
of each LLM instance's counters across the run. Reading those counters as
absolute totals instead both multiplied usage across agents sharing an
instance and carried earlier runs into later ones.

Baselines are taken per distinct instance rather than per agent, so a shared
instance is measured once and no per-agent window exists to overlap when tasks
run concurrently.
"""

from __future__ import annotations

from crewai import Agent, Crew, Task, LLM
from crewai.types.usage_metrics import UsageMetrics


class _Counter:
    """Stands in for an LLM instance's cumulative lifetime counters."""

    def __init__(self) -> None:
        self.total = 0

    def bind(self, llm: LLM) -> LLM:
        llm.get_token_usage_summary = lambda: UsageMetrics(  # type: ignore[method-assign]
            total_tokens=self.total,
            prompt_tokens=self.total,
            successful_requests=1 if self.total else 0,
        )
        return llm

    def consume(self, tokens: int) -> None:
        self.total += tokens


def _crew(*llms: LLM) -> Crew:
    agents = [
        Agent(role=f"Role {i}", goal="goal", backstory="backstory", llm=llm)
        for i, llm in enumerate(llms)
    ]
    tasks = [
        Task(description=f"task {i}", expected_output="out", agent=agent)
        for i, agent in enumerate(agents)
    ]
    return Crew(agents=agents, tasks=tasks)


def test_agents_sharing_one_llm_are_counted_once() -> None:
    """Three agents on one instance report that instance's usage, not 3x it."""
    counter = _Counter()
    llm = counter.bind(LLM(model="gpt-4o"))
    crew = _crew(llm, llm, llm)

    crew._snapshot_usage_baselines()
    counter.consume(100)

    assert crew.calculate_usage_metrics().total_tokens == 100


def test_distinct_llm_instances_are_summed() -> None:
    """Separate instances still add up, even for the same model."""
    first, second = _Counter(), _Counter()
    crew = _crew(first.bind(LLM(model="gpt-4o")), second.bind(LLM(model="gpt-4o")))

    crew._snapshot_usage_baselines()
    first.consume(100)
    second.consume(50)

    assert crew.calculate_usage_metrics().total_tokens == 150


def test_a_later_run_excludes_the_previous_one() -> None:
    """Usage is the growth across this run, not the instance's lifetime."""
    counter = _Counter()
    crew = _crew(counter.bind(LLM(model="gpt-4o")))

    crew._snapshot_usage_baselines()
    counter.consume(100)
    assert crew.calculate_usage_metrics().total_tokens == 100

    crew._snapshot_usage_baselines()
    counter.consume(50)
    assert crew.calculate_usage_metrics().total_tokens == 50


def test_manager_sharing_an_agent_llm_is_counted_once() -> None:
    """A manager on the same instance as its agents adds no extra usage."""
    counter = _Counter()
    llm = counter.bind(LLM(model="gpt-4o"))
    crew = _crew(llm)
    crew.manager_agent = Agent(
        role="Manager", goal="coordinate", backstory="backstory", llm=llm
    )

    crew._snapshot_usage_baselines()
    counter.consume(100)

    assert crew.calculate_usage_metrics().total_tokens == 100
