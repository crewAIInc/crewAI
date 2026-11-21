from __future__ import annotations

from typing import Any

import pytest

from crewai.llms.base_llm import BaseLLM


class StubLLM(BaseLLM):
    """Minimal concrete LLM for exercising shared BaseLLM behavior."""

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
        return ""

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
        raise NotImplementedError

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 4096


@pytest.mark.parametrize("stop_words", [["LATE", "EARLY"], ["EARLY", "LATE"]])
def test_apply_stop_words_truncates_at_earliest_match(
    stop_words: list[str],
) -> None:
    llm = StubLLM(model="test-model", stop=stop_words)

    assert llm._apply_stop_words("hello EARLY middle LATE tail") == "hello"


def test_apply_stop_words_keeps_text_when_no_stop_word_matches() -> None:
    llm = StubLLM(model="test-model", stop=["NOPE"])
    content = "hello world"

    assert llm._apply_stop_words(content) == content


def test_apply_stop_words_keeps_text_when_stop_list_is_empty() -> None:
    llm = StubLLM(model="test-model", stop=[])
    content = "hello world"

    assert llm._apply_stop_words(content) == content


def test_apply_stop_words_keeps_empty_content_empty() -> None:
    llm = StubLLM(model="test-model", stop=["Observation:"])

    assert llm._apply_stop_words("") == ""

