"""Smoke checks for BaseLLM._apply_stop_words."""

from typing import Any

from crewai.llms.base_llm import BaseLLM


class _StubLLM(BaseLLM):
    def call(self, *args: Any, **kwargs: Any) -> str:
        return ""


def test_apply_stop_words_truncates_at_the_earliest_text_match() -> None:
    """Both stop-list orders cut at the earliest match and strip the cut."""
    content = "hello EARLY middle LATE tail"
    late_then_early = _StubLLM(model="stub", stop=["LATE", "EARLY"])
    early_then_late = _StubLLM(model="stub", stop=["EARLY", "LATE"])

    assert late_then_early._apply_stop_words(content) == "hello"
    assert early_then_late._apply_stop_words(content) == "hello"
    assert late_then_early._apply_stop_words("  hello \nEARLY tail") == "hello"


def test_apply_stop_words_leaves_text_unchanged_when_nothing_matches() -> None:
    llm = _StubLLM(model="stub", stop=["LATE", "EARLY"])
    unchanged = "no stop words here"

    assert llm._apply_stop_words(unchanged) == unchanged
    assert _StubLLM(model="stub")._apply_stop_words(unchanged) == unchanged
