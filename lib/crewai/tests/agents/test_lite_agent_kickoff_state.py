"""LiteAgent kickoff must not leak state from previous kickoffs.

Two instance attributes survived from one kickoff into the next:

- ``_guardrail_retry_count`` was never reset, so after one run exhausted the
  retry budget, every later kickoff raised "guardrail failed validation
  after N retries" on its *first* guardrail failure with zero retries.
  ``Agent``'s guardrail loop uses a local counter per kickoff; LiteAgent is
  the asymmetrical twin.
- ``usage_metrics`` reported the LLM instance's lifetime cumulative
  counters (``get_token_usage_summary()``) instead of the per-call delta.
  ``Agent.kickoff`` reports ``delta_since`` a baseline snapshot (see
  ``TestKickoffUsageMetricsArePerCall``); the LiteAgent twin never got that
  fix, so repeated kickoffs (or agents sharing an LLM) double-counted every
  previous call's tokens.
"""

import warnings

import pytest

from crewai.lite_agent import LiteAgent
from crewai.llms.base_llm import BaseLLM


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


def _make_agent(**kwargs) -> LiteAgent:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return LiteAgent(
            role="Researcher",
            goal="Answer questions.",
            backstory="Test agent.",
            llm=_FixedUsageLLM(),
            **kwargs,
        )


@pytest.fixture(autouse=True)
def _ignore_deprecation():
    warnings.simplefilter("ignore", FutureWarning)


def test_repeated_kickoffs_report_per_call_usage():
    agent = _make_agent()
    r1 = agent.kickoff("question one")
    r2 = agent.kickoff("question two")

    assert r1.usage_metrics is not None
    assert r1.usage_metrics["prompt_tokens"] == 100
    # The second kickoff must report only its own call's usage, matching
    # Agent.kickoff semantics (see TestKickoffUsageMetricsArePerCall).
    assert r2.usage_metrics == r1.usage_metrics


def test_guardrail_retry_budget_resets_between_kickoffs():
    calls = {"n": 0}

    def always_fail(output):
        calls["n"] += 1
        return (False, "reject")

    agent = _make_agent(guardrail=always_fail, guardrail_max_retries=3)

    # Kickoff 1 exhausts the full retry budget (4 evaluations: 1 + 3 retries)
    # and raises.
    with pytest.raises(Exception, match="guardrail failed validation"):
        agent.kickoff("first question")
    assert calls["n"] == 4

    # Kickoff 2 is a fresh run on the same agent instance: it must get the
    # full retry budget again (4 evaluations), not inherit kickoff 1's count.
    calls["n"] = 0
    with pytest.raises(Exception, match="guardrail failed validation"):
        agent.kickoff("second question")
    assert calls["n"] == 4, (
        f"expected 4 guardrail evaluations in the second kickoff, got {calls['n']}"
    )
