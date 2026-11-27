# mypy: ignore-errors
"""Smoke test for the offline custom LLM quick-start example.

Verifies the OfflineLLM stub documented in docs/edge/en/learn/custom-llm.mdx
runs a complete Agent/Task/Crew cycle without any provider API key.

Fixes #8003
"""

from crewai import Agent, BaseLLM, Crew, Task


class OfflineLLM(BaseLLM):
    """Deterministic LLM stub that needs no network or API key."""

    def __init__(self):
        super().__init__(model="offline-stub")

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
        return "Thought: I know the answer.\nFinal Answer: Hello from OfflineLLM!"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096


class TestOfflineLLMQuickstart:
    """The documented offline example must work end-to-end."""

    def test_direct_call_returns_deterministic_response(self):
        llm = OfflineLLM()
        result = llm.call("Hi")
        assert "Hello from OfflineLLM!" in result

    def test_crew_kickoff_returns_expected_output(self):
        llm = OfflineLLM()

        agent = Agent(
            role="Greeter",
            goal="Say hello",
            backstory="A friendly offline agent.",
            llm=llm,
        )

        task = Task(
            description="Greet the user",
            expected_output="A short greeting",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert "Hello from OfflineLLM!" in result.raw
