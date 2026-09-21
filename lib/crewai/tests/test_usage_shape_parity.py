# mypy: ignore-errors
"""Regression tests for EPD-178: token usage was exposed in different shapes
and attribute names per code path — ``Agent.kickoff()`` results carried a
plain dict at ``.usage_metrics`` (no ``token_usage`` attribute at all), while
``Crew.kickoff()`` results carried a ``UsageMetrics`` object at
``.token_usage`` (no ``usage_metrics`` attribute), so any single accessor
written for one path raised ``AttributeError`` on the other.

Both result types now expose both surfaces: ``.token_usage`` as a
``UsageMetrics`` object and ``.usage_metrics`` as a plain dict.
"""

from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llms.base_llm import BaseLLM
from crewai.types.usage_metrics import UsageMetrics


class _FixedUsageLLM(BaseLLM):
    """Offline BaseLLM that records fixed usage (100/10 tokens) per call."""

    def __init__(self):
        super().__init__(model="fixed-usage-model")

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
        return "Thought: I know the answer.\nFinal Answer: fake answer"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096


class TestUsageShapeUnitParity:
    def test_lite_agent_output_exposes_token_usage_object(self):
        metrics = UsageMetrics(
            total_tokens=110,
            prompt_tokens=100,
            completion_tokens=10,
            successful_requests=1,
        )
        output = LiteAgentOutput(
            agent_role="analyst", usage_metrics=metrics.model_dump()
        )

        assert output.token_usage == metrics
        assert isinstance(output.token_usage, UsageMetrics)

    def test_lite_agent_output_token_usage_zeroed_when_absent(self):
        output = LiteAgentOutput(agent_role="analyst")

        assert output.usage_metrics is None
        assert output.token_usage == UsageMetrics()

    def test_crew_output_exposes_usage_metrics_dict(self):
        metrics = UsageMetrics(
            total_tokens=110,
            prompt_tokens=100,
            completion_tokens=10,
            successful_requests=1,
        )
        output = CrewOutput(token_usage=metrics)

        assert output.usage_metrics == metrics.model_dump()
        assert isinstance(output.usage_metrics, dict)

    def test_both_shapes_carry_identical_keys(self):
        """The dict shape has exactly the UsageMetrics fields on both types."""
        crew_dict = CrewOutput(token_usage=UsageMetrics()).usage_metrics
        lite = LiteAgentOutput(
            agent_role="analyst", usage_metrics=UsageMetrics().model_dump()
        )

        assert set(crew_dict) == set(UsageMetrics.model_fields)
        assert set(lite.usage_metrics) == set(UsageMetrics.model_fields)


class TestUsageShapeEndToEnd:
    """Mirror of the EPD-178 clean-room repro, offline via a fake BaseLLM."""

    @staticmethod
    def _read_via_object(result) -> int:
        """Single accessor written against the CrewOutput shape."""
        return result.token_usage.prompt_tokens

    @staticmethod
    def _read_via_dict(result) -> int:
        """Single accessor written against the LiteAgentOutput shape."""
        return result.usage_metrics["prompt_tokens"]

    def test_single_accessor_works_on_both_kickoff_paths(self):
        agent_a = Agent(
            role="analyst",
            goal="Answer questions.",
            backstory="Test agent.",
            llm=_FixedUsageLLM(),
            verbose=False,
        )
        result_agent = agent_a.kickoff("a question")

        agent_b = Agent(
            role="analyst",
            goal="Answer questions.",
            backstory="Test agent.",
            llm=_FixedUsageLLM(),
            verbose=False,
        )
        task = Task(
            description="Answer: a question",
            expected_output="A short answer.",
            agent=agent_b,
        )
        crew = Crew(agents=[agent_b], tasks=[task], verbose=False)
        result_crew = crew.kickoff()

        assert isinstance(result_agent, LiteAgentOutput)
        assert isinstance(result_crew, CrewOutput)

        # Both accessors work on both result types and agree with each other.
        for result in (result_agent, result_crew):
            object_read = self._read_via_object(result)
            dict_read = self._read_via_dict(result)
            assert object_read == dict_read
            assert object_read > 0
            assert isinstance(result.token_usage, UsageMetrics)
            assert isinstance(result.usage_metrics, dict)
