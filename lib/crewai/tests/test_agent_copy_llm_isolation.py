# mypy: ignore-errors
"""Regression test for #9587: Agent.copy() shares LLM mutable dicts.

copy.copy creates a new LLM object but additional_params and _token_usage
stay shared with the original. Mutating them on the copy changes the original.
"""

from crewai import Agent
from crewai.llms.base_llm import BaseLLM


class _StubLLM(BaseLLM):
    """Offline LLM for testing."""

    def __init__(self):
        super().__init__(model="stub-model")

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
        return "Thought: I know the answer.\nFinal Answer: done"


class TestAgentCopyLLMIsolation:
    """Agent.copy() must isolate LLM mutable state."""

    def test_additional_params_not_shared(self):
        llm = _StubLLM()
        llm.additional_params["bucket"] = []

        original = Agent(
            role="original",
            goal="test",
            backstory="test",
            llm=llm,
        )

        copied = original.copy()

        copied.llm.additional_params["bucket"].append(1)

        assert original.llm.additional_params["bucket"] == [], (
            "Original agent's additional_params was mutated by the copy"
        )

    def test_token_usage_not_shared(self):
        llm = _StubLLM()

        original = Agent(
            role="original",
            goal="test",
            backstory="test",
            llm=llm,
        )

        copied = original.copy()

        copied.llm._token_usage["total_tokens"] += 100

        assert original.llm._token_usage["total_tokens"] == 0, (
            "Original agent's _token_usage was mutated by the copy"
        )

    def test_copy_is_separate_llm_instance(self):
        llm = _StubLLM()

        original = Agent(
            role="original",
            goal="test",
            backstory="test",
            llm=llm,
        )

        copied = original.copy()

        assert copied.llm is not original.llm, (
            "Agent.copy() should create a separate LLM instance"
        )
