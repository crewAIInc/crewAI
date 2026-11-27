"""The documented manual stop-word filter stops at the earliest match, in either order."""

from types import SimpleNamespace

from crewai.llms.base_llm import BaseLLM


def _documented_filter(content: str, stops: list[str]) -> str:
    """The snippet from docs/edge/*/learn/custom-llm.mdx, "Stop Words Support"."""
    if stops:
        positions = [pos for pos in (content.find(word) for word in stops) if pos != -1]
        if positions:
            content = content[: min(positions)].strip()
    return content


def test_documented_filter_stops_at_the_earliest_match_regardless_of_order():
    content = "hello EARLY middle LATE tail"

    assert _documented_filter(content, ["LATE", "EARLY"]) == "hello"
    assert _documented_filter(content, ["EARLY", "LATE"]) == "hello"


def test_documented_filter_leaves_content_without_a_match_alone():
    content = "hello world"

    assert _documented_filter(content, ["EARLY", "LATE"]) == content


def test_documented_filter_agrees_with_the_runtime_helper():
    content = "hello EARLY middle LATE tail"

    for stops in (["LATE", "EARLY"], ["EARLY", "LATE"]):
        stub = SimpleNamespace(stop_sequences=stops)
        assert _documented_filter(content, stops) == BaseLLM._apply_stop_words(stub, content)
