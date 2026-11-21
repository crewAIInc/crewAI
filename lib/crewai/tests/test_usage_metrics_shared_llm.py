# mypy: ignore-errors
"""Regression test for #7259: usage_metrics multiplied token usage
when multiple agents shared the same LLM instance.

calculate_usage_metrics() iterated over agents and called
llm.get_token_usage_summary() for each one. When N agents shared
one LLM instance, that instance's cumulative totals were added N times,
inflating reported usage by Nx.
"""

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.types.usage_metrics import UsageMetrics


class _FixedLLM(BaseLLM):
    """Offline BaseLLM that records fixed token usage per call."""

    def __init__(self):
        super().__init__(model="fixed-model")

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        self._track_token_usage_internal(
            {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        )
        return "Thought: I know the answer.\nFinal Answer: done"


class TestSharedLLMUsageDedup:
    """When agents share an LLM instance, usage must be counted once."""

    def test_shared_llm_not_double_counted(self):
        shared_llm = _FixedLLM()

        agent_a = Agent(
            role="Alpha",
            goal="Do alpha things",
            backstory="Alpha agent",
            llm=shared_llm,
        )
        agent_b = Agent(
            role="Beta",
            goal="Do beta things",
            backstory="Beta agent",
            llm=shared_llm,
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent_a,
        )

        crew = Crew(agents=[agent_a, agent_b], tasks=[task])

        # Simulate: the shared LLM was called once (110 total tokens)
        shared_llm._track_token_usage_internal(
            {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        )

        metrics = crew.calculate_usage_metrics()

        # Before fix: 110 * 2 agents = 220 (wrong)
        # After fix:  110 * 1 (deduped) = 110 (correct)
        assert metrics.total_tokens == 110, (
            f"Expected 110 total_tokens (counted once), got {metrics.total_tokens}. "
            f"Shared LLM usage was double-counted."
        )

    def test_separate_llms_counted_independently(self):
        llm_a = _FixedLLM()
        llm_b = _FixedLLM()

        agent_a = Agent(
            role="Alpha",
            goal="Do alpha things",
            backstory="Alpha agent",
            llm=llm_a,
        )
        agent_b = Agent(
            role="Beta",
            goal="Do beta things",
            backstory="Beta agent",
            llm=llm_b,
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent_a,
        )

        crew = Crew(agents=[agent_a, agent_b], tasks=[task])

        llm_a._track_token_usage_internal(
            {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        )
        llm_b._track_token_usage_internal(
            {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}
        )

        metrics = crew.calculate_usage_metrics()

        # Separate LLMs: 110 + 220 = 330
        assert metrics.total_tokens == 330, (
            f"Expected 330 total_tokens (separate LLMs summed), got {metrics.total_tokens}"
        )

    def test_three_agents_one_llm(self):
        shared_llm = _FixedLLM()

        agents = [
            Agent(role=f"Agent{i}", goal="test", backstory="test", llm=shared_llm)
            for i in range(3)
        ]

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agents[0],
        )

        crew = Crew(agents=agents, tasks=[task])

        shared_llm._track_token_usage_internal(
            {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}
        )

        metrics = crew.calculate_usage_metrics()

        # Before fix: 55 * 3 = 165 (wrong)
        # After fix:  55 * 1 = 55 (correct)
        assert metrics.total_tokens == 55, (
            f"Expected 55 total_tokens with 3 agents sharing 1 LLM, got {metrics.total_tokens}"
        )
