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

def test_new_id_without_name_gets_a_new_slot() -> None:
    calls: dict[int, dict[str, object]] = {
        0: {"id": "call_paris", "name": "get_weather", "arguments": ""}
    }

    assert _tool_call_index(None, "call_london", None, calls) == 1


def test_name_without_id_completes_latest_id_without_name() -> None:
    calls: dict[int, dict[str, object]] = {
        0: {"id": "call_london", "name": "", "arguments": ""}
    }

    assert _tool_call_index(None, None, "get_weather", calls) == 0


def test_name_without_id_starts_new_slot_when_no_call_is_pending() -> None:
    calls: dict[int, dict[str, object]] = {
        0: {"id": "call_paris", "name": "get_weather", "arguments": ""}
    }

    assert _tool_call_index(None, None, "get_weather", calls) == 1
