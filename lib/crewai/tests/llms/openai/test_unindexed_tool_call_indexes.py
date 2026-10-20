"""Tests for OpenAI-compatible streams that omit tool-call indexes."""

from crewai.llms.providers.openai.completion import _tool_call_index


def test_unindexed_calls_with_distinct_ids_get_distinct_slots() -> None:
    calls: dict[int, dict[str, object]] = {}
    first = _tool_call_index(None, "call_paris", "get_weather", calls)
    calls[first] = {"id": "call_paris", "name": "get_weather", "arguments": ""}
    second = _tool_call_index(None, "call_london", "get_weather", calls)

    assert (first, second) == (0, 1)


def test_repeated_id_and_argument_only_delta_continue_latest_call() -> None:
    calls: dict[int, dict[str, object]] = {
        0: {"id": "call_paris", "name": "get_weather", "arguments": ""}
    }

    assert _tool_call_index(None, "call_paris", None, calls) == 0
    assert _tool_call_index(None, None, None, calls) == 0


def test_explicit_index_is_preserved() -> None:
    calls: dict[int, dict[str, object]] = {
        0: {"id": "call_paris", "name": "get_weather", "arguments": ""}
    }

    assert _tool_call_index(7, "call_london", "get_weather", calls) == 7
