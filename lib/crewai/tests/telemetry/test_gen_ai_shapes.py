"""Unit tests for the OTel GenAI parts-shape transforms."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

import json

from crewai.telemetry.tracing.gen_ai_shapes import (
    DEFAULT_MAX_ATTR_BYTES,
    coerce_finish_reason,
    finish_reasons_from_messages,
    to_input_messages,
    to_output_messages,
    to_system_instructions,
    to_tool_call_arguments,
    to_tool_call_result,
    to_tool_definitions,
    truncate_attr,
)


class _Args(BaseModel):
    query: str = Field(description="search query")


class _FakeTool:
    name = "search"
    description = "search the web"
    args_schema = _Args


@pytest.mark.parametrize("empty", [None, "", []])
def test_input_and_output_messages_drop_empty_inputs(empty: object) -> None:
    assert to_input_messages(empty) is None
    assert to_output_messages(empty) is None


def test_input_messages_translates_openai_dicts_to_parts_shape() -> None:
    result = to_input_messages(
        [
            {"role": "system", "content": "be helpful"},
            {
                "role": "assistant",
                "content": "calling search",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "search", "arguments": '{"q":"x"}'}}
                ],
            },
        ]
    )
    assert result == [
        {"role": "system", "parts": [{"type": "text", "content": "be helpful"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "calling search"},
                {"type": "tool_call", "name": "search", "arguments": {"q": "x"}, "id": "c1"},
            ],
        },
    ]


def test_input_messages_accepts_string_and_bare_dict() -> None:
    assert to_input_messages("hi") == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert to_input_messages({"role": "system", "content": "be brief"}) == [
        {"role": "system", "parts": [{"type": "text", "content": "be brief"}]}
    ]


def test_output_messages_string_response_emits_assistant_with_finish_stop() -> None:
    assert to_output_messages("done") == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "done"}],
            "finish_reason": "stop",
        }
    ]


def test_output_messages_pure_tool_call_list_uses_tool_call_finish() -> None:
    result = to_output_messages(
        [{"id": "1", "function": {"name": "search", "arguments": "{}"}}]
    )
    assert result == [
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "name": "search", "arguments": {}, "id": "1"}],
            "finish_reason": "tool_call",
        }
    ]


def test_output_messages_aliases_provider_finish_reasons_to_spec_enum() -> None:
    # Anthropic emits `end_turn` rather than spec's `stop`.
    result = to_output_messages({"role": "assistant", "content": "x", "finish_reason": "end_turn"})
    assert result is not None and result[0]["finish_reason"] == "stop"


def test_tool_definitions_extracts_basetool_args_schema() -> None:
    # `args_schema` is a Pydantic class — without explicit handling it would
    # be `repr()`-stringified by the default serializer, losing the schema.
    result = to_tool_definitions([_FakeTool()])
    assert result is not None and result[0]["name"] == "search"
    assert "properties" in result[0]["parameters"]


def test_tool_definitions_hoists_openai_wrapped_function_to_flat() -> None:
    wrapped = {"type": "function", "function": {"name": "x", "description": "d"}}
    assert to_tool_definitions([wrapped]) == [{"type": "function", "name": "x", "description": "d"}]


def test_tool_call_arguments_parses_json_string_falls_back_to_raw() -> None:
    assert to_tool_call_arguments('{"q": "rain"}') == {"q": "rain"}
    assert to_tool_call_arguments("not-json") == "not-json"


def test_tool_call_result_parses_json_string_falls_back_to_raw() -> None:
    assert to_tool_call_result('{"temperature": 57}') == {"temperature": 57}
    assert to_tool_call_result("hello") == "hello"


def test_system_instructions_wraps_string_in_text_part() -> None:
    assert to_system_instructions("Be helpful") == [{"type": "text", "content": "Be helpful"}]


def test_finish_reasons_from_messages_skips_messages_without_a_value() -> None:
    messages = [{"role": "assistant"}, {"role": "assistant", "finish_reason": "length"}]
    assert finish_reasons_from_messages(messages) == ["length"]

_FINISH_REASON_CASES = [
    # → stop
    ("STOP", "stop"),
    ("stop", "stop"),
    ("stop_sequence", "stop"),
    ("pause_turn", "stop"),
    ("end_turn", "stop"),
    ("completed", "stop"),  # OpenAI Responses API
    ("Completed", "stop"),  # case-insensitivity sanity
    ("COMPLETED", "stop"),
    # → length
    ("MAX_TOKENS", "length"),
    ("max_tokens", "length"),
    ("model_context_window_exceeded", "length"),
    ("length", "length"),
    ("incomplete", "length"),  # OpenAI Responses API (lossy bucket)
    # → content_filter
    ("SAFETY", "content_filter"),
    ("BLOCKLIST", "content_filter"),
    ("PROHIBITED_CONTENT", "content_filter"),
    ("SPII", "content_filter"),
    ("MODEL_ARMOR", "content_filter"),
    ("IMAGE_SAFETY", "content_filter"),
    ("IMAGE_PROHIBITED_CONTENT", "content_filter"),
    ("refusal", "content_filter"),
    ("content_filtered", "content_filter"),
    ("guardrail_intervened", "content_filter"),
    ("content_filter", "content_filter"),
    # → tool_call
    ("tool_use", "tool_call"),
    ("tool_calls", "tool_call"),
    ("function_call", "tool_call"),
    ("tool_call", "tool_call"),
    # → error
    ("RECITATION", "error"),
    ("IMAGE_RECITATION", "error"),
    ("MALFORMED_FUNCTION_CALL", "error"),
    ("OTHER", "error"),
    ("IMAGE_OTHER", "error"),
    ("LANGUAGE", "error"),
    ("error", "error"),
    ("failed", "error"),  # OpenAI Responses API
    ("cancelled", "error"),  # OpenAI Responses API
]


@pytest.mark.parametrize(("raw", "expected"), _FINISH_REASON_CASES)
def test_coerce_finish_reason_maps_provider_values_to_otel_enum(
    raw: str, expected: str
) -> None:
    assert coerce_finish_reason(raw) == expected


@pytest.mark.parametrize(
    "empty",
    [
        None,
        "",
        "FINISH_REASON_UNSPECIFIED",
        "finish_reason_unspecified",
        # OpenAI Responses API lifecycle states — not real finish reasons.
        "in_progress",
        "IN_PROGRESS",  # case-insensitivity sanity
        "queued",
    ],
)
def test_coerce_finish_reason_drops_empty_and_unspecified(empty: object) -> None:
    assert coerce_finish_reason(empty) is None  # type: ignore[arg-type]


def test_coerce_finish_reason_flags_unknown_value_as_error() -> None:
    assert coerce_finish_reason("totally_unknown_value") == "error"


@pytest.mark.parametrize(
    "raw_finish",
    [
        pytest.param({"role": "assistant", "content": "hi"}, id="finish_key_missing"),
        pytest.param(
            {"role": "assistant", "content": "hi", "finish_reason": ""},
            id="finish_empty_string",
        ),
        pytest.param(
            {
                "role": "assistant",
                "content": "hi",
                "finish_reason": "FINISH_REASON_UNSPECIFIED",
            },
            id="finish_gemini_unspecified_sentinel",
        ),
        pytest.param(
            {"role": "assistant", "content": "hi", "finish_reason": "in_progress"},
            id="finish_openai_responses_in_progress",
        ),
        pytest.param(
            {"role": "assistant", "content": "hi", "finish_reason": "queued"},
            id="finish_openai_responses_queued",
        ),
    ],
)
def test_output_messages_inference_path_falls_back_to_stop_for_missing_finish(
    raw_finish: dict,
) -> None:
    # The internal `_coerce_finish` helper is exercised through its only
    # caller, `to_output_messages`. Missing / blank / unspecified values, plus
    # OpenAI Responses API lifecycle states that aren't real finish reasons,
    # must resolve to the default `"stop"` rather than being flagged as
    # unknown.
    result = to_output_messages(raw_finish)
    assert result is not None and result[0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", "stop"),
        ("incomplete", "length"),
        ("failed", "error"),
        ("cancelled", "error"),
    ],
)
def test_output_messages_aliases_openai_responses_api_status_values(
    status: str, expected: str
) -> None:
    # Covers `_coerce_finish` (inference path) for the OpenAI Responses API
    # status values — complements the `coerce_finish_reason` (explicit-finish
    # path) coverage in `_FINISH_REASON_CASES`.
    result = to_output_messages(
        {"role": "assistant", "content": "hi", "finish_reason": status}
    )
    assert result is not None and result[0]["finish_reason"] == expected


def test_truncate_attr_passes_through_payload_under_cap() -> None:
    payload = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": "small"}]}]
    )

    result, markers = truncate_attr(payload, attr="gen_ai.input.messages")

    assert result == payload
    assert markers == {}


def test_truncate_attr_returns_none_for_none_input() -> None:
    assert truncate_attr(None, attr="gen_ai.input.messages") == (None, {})


def test_truncate_attr_message_array_drops_middle_with_placeholder_when_over_cap() -> (
    None
):
    body = "x" * 4_096
    messages = [
        {"role": f"user-{i}", "parts": [{"type": "text", "content": body}]}
        for i in range(10)
    ]
    payload = json.dumps(messages)
    cap = 16 * 1024

    result, markers = truncate_attr(
        payload, attr="gen_ai.input.messages", max_bytes=cap
    )

    assert markers == {
        "gen_ai.input.messages.truncated": True,
        "gen_ai.input.messages.original_size_bytes": len(payload.encode("utf-8")),
    }
    parsed = json.loads(result)
    assert isinstance(parsed, list) and len(parsed) == 3
    assert parsed[0]["role"] == "user-0"
    assert parsed[-1]["role"] == "user-9"
    placeholder_text = parsed[1]["parts"][0]["content"]
    assert placeholder_text.startswith("[truncated 8 messages")
    assert len(result.encode("utf-8")) <= cap


def test_truncate_attr_message_array_with_one_giant_message_head_tail_truncates_content() -> (
    None
):
    huge = "A" * 60_000 + "Z" * 60_000
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": huge}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "ok"}]},
    ]
    payload = json.dumps(messages)
    cap = 8 * 1024

    result, markers = truncate_attr(
        payload, attr="gen_ai.output.messages", max_bytes=cap
    )

    assert markers["gen_ai.output.messages.truncated"] is True
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    user_content = parsed[0]["parts"][0]["content"]
    assert "[truncated" in user_content and "KB]" in user_content
    assert user_content.startswith("A")
    assert user_content.endswith("Z")
    assert len(result.encode("utf-8")) <= cap


def test_truncate_attr_non_message_json_wraps_in_envelope_with_preview() -> None:
    payload = json.dumps({"city": "SF", "hits": "x" * 50_000})
    cap = 4 * 1024

    result, markers = truncate_attr(
        payload, attr="gen_ai.tool.call.arguments", max_bytes=cap
    )

    assert markers["gen_ai.tool.call.arguments.truncated"] is True
    parsed = json.loads(result)
    assert parsed["_truncated"] is True
    assert parsed["_original_size_bytes"] == len(payload.encode("utf-8"))
    assert isinstance(parsed["_preview"], str) and parsed["_preview"]
    assert payload.startswith(parsed["_preview"])


def test_truncate_attr_malformed_json_falls_back_to_envelope() -> None:
    payload = "[not json " * 5_000
    cap = 1_024

    result, markers = truncate_attr(
        payload, attr="gen_ai.input.messages", max_bytes=cap
    )

    assert markers["gen_ai.input.messages.truncated"] is True
    parsed = json.loads(result)
    assert parsed["_truncated"] is True
    assert parsed["_original_size_bytes"] == len(payload.encode("utf-8"))


def test_truncate_attr_env_override_overrides_default_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": "small"}]}]
    )
    assert len(payload.encode("utf-8")) < DEFAULT_MAX_ATTR_BYTES

    monkeypatch.setenv("CREWAI_OTEL_MAX_ATTR_BYTES", "16")
    result, markers = truncate_attr(payload, attr="gen_ai.input.messages")

    assert markers["gen_ai.input.messages.truncated"] is True
    assert markers["gen_ai.input.messages.original_size_bytes"] == len(
        payload.encode("utf-8")
    )
    json.loads(result)


def test_truncate_attr_env_override_invalid_value_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        [{"role": "user", "parts": [{"type": "text", "content": "small"}]}]
    )
    monkeypatch.setenv("CREWAI_OTEL_MAX_ATTR_BYTES", "not-a-number")

    result, markers = truncate_attr(payload, attr="gen_ai.input.messages")

    assert result == payload
    assert markers == {}


def test_truncate_attr_preserves_utf8_in_head_tail_truncation() -> None:
    # Multi-byte chars must not get sliced mid-character; decode("utf-8",
    # errors="ignore") drops dangling bytes so the result stays valid.
    body = "日本語" * 20_000  # 3-byte chars repeated → ~180KB
    messages = [
        {"role": "user", "parts": [{"type": "text", "content": body}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "ok"}]},
    ]
    payload = json.dumps(messages, ensure_ascii=False)
    cap = 8 * 1024

    result, _ = truncate_attr(payload, attr="gen_ai.input.messages", max_bytes=cap)

    parsed = json.loads(result)
    assert isinstance(parsed, list)
    json.dumps(parsed, ensure_ascii=False)  # round-trips, i.e. no surrogates
    assert len(result.encode("utf-8")) <= cap
