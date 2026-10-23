"""Guardrail retries must show the model why the previous answer failed."""

from __future__ import annotations

from typing import Any

from crewai import Agent
from crewai.llms.base_llm import BaseLLM


class _ScriptedLLM(BaseLLM):
    """Return scripted ReAct answers and record each message list."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(model="scripted")
        object.__setattr__(self, "_responses", responses)
        object.__setattr__(self, "seen", [])

    def call(self, messages: Any, **kwargs: Any) -> str:
        self.seen.append([dict(message) for message in messages])
        index = min(len(self.seen) - 1, len(self._responses) - 1)
        return self._responses[index]

    async def acall(self, messages: Any, **kwargs: Any) -> str:
        return self.call(messages, **kwargs)

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


def _text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content") or "") for message in messages)


def _guardrail(output: Any) -> tuple[bool, Any]:
    if "MAGIC" in output.raw:
        return True, output
    return False, "The answer must contain MAGIC."


def test_guardrail_retry_includes_the_validation_feedback() -> None:
    """The retry used to reset the conversation and re-send the first prompt."""
    llm = _ScriptedLLM(
        [
            "Thought: answer directly.\nFinal Answer: wrong answer",
            "Thought: correct it.\nFinal Answer: MAGIC",
        ]
    )
    agent = Agent(
        role="tester",
        goal="answer",
        backstory="test",
        llm=llm,
        guardrail=_guardrail,
        guardrail_max_retries=1,
        verbose=False,
    )

    result = agent.kickoff("Answer the question.")

    assert result.raw == "MAGIC"
    assert len(llm.seen) == 2
    assert "must contain MAGIC" not in _text(llm.seen[0])
    retry_text = _text(llm.seen[1])
    assert "must contain MAGIC" in retry_text
    assert "wrong answer" in retry_text
    assert llm.seen[1][-1]["role"] == "user"
