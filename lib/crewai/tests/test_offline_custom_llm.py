"""Smoke test for the runnable offline custom LLM example in the custom-llm guide.

Mirrors the ``OfflineLLM`` shown in ``docs/edge/en/learn/custom-llm.mdx`` and checks its
direct-call behaviour with no network or API key. The full Agent/Task/Crew run is covered by
``test_custom_llm_within_crew`` in ``test_custom_llm.py``.
"""

from crewai.llms.base_llm import BaseLLM


class OfflineLLM(BaseLLM):
    """A deterministic, offline LLM — no API key, no network. For learning and tests."""

    def __init__(self, response: str = "Hello from your custom LLM!", model: str = "offline-stub"):
        super().__init__(model=model)
        self.response = response

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
        if "Thought:" in str(messages):
            return f"Thought: I now know the final answer\nFinal Answer: {self.response}"
        return self.response

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False


def test_offline_llm_returns_its_fixed_response():
    llm = OfflineLLM(response="42")
    assert llm.call("What is the answer to everything?") == "42"


def test_offline_llm_emits_a_final_answer_for_the_react_loop():
    llm = OfflineLLM(response="Hello! Nice to meet you!")
    result = llm.call([{"role": "user", "content": "Thought: I should greet the user"}])
    assert "Final Answer: Hello! Nice to meet you!" in result


def test_offline_llm_declares_capabilities_honestly():
    llm = OfflineLLM()
    assert llm.supports_function_calling() is False
    assert llm.supports_stop_words() is False
