"""Boundary tests for `semantic_conventions.gen_ai`.

`gen_ai()` owns both the OTel attribute keys and the value shapes for the
GenAI spec. It runs `gen_ai_shapes` transforms on raw event values and
JSON-encodes the results. Any failure must drop the attribute and log,
never raise — instrumentation sits on the user's request path.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import patch

import pytest

from crewai.telemetry.tracing import semantic_conventions


def test_gen_ai_shapes_raw_values_into_spec_attrs() -> None:
    attrs = semantic_conventions.gen_ai(
        operation_name="chat",
        system_instructions="be helpful",
        input_messages=[{"role": "user", "content": "hi"}],
        output_messages="done",
    )

    assert attrs["gen_ai.operation.name"] == "chat"
    assert json.loads(attrs["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "be helpful"}
    ]
    assert json.loads(attrs["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert json.loads(attrs["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "done"}],
            "finish_reason": "stop",
        }
    ]
    # finish_reasons is a raw list, not a JSON string — OTel SDK serializes
    # list-of-string attributes natively for `gen_ai.response.finish_reasons`.
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


def test_gen_ai_drops_none_values() -> None:
    attrs = semantic_conventions.gen_ai(operation_name="chat")
    assert attrs == {"gen_ai.operation.name": "chat"}


def test_gen_ai_emits_conversation_id_when_provided_and_drops_when_omitted() -> None:
    with_id = semantic_conventions.gen_ai(
        operation_name="chat", conversation_id="conv-1"
    )
    without_id = semantic_conventions.gen_ai(operation_name="chat")

    assert with_id["gen_ai.conversation.id"] == "conv-1"
    assert "gen_ai.conversation.id" not in without_id


def test_gen_ai_emits_workflow_name_for_orchestration_spans() -> None:
    attrs = semantic_conventions.gen_ai(
        operation_name=semantic_conventions.GEN_AI_OP_INVOKE_WORKFLOW,
        workflow_name="ContentApprovalFlow",
    )

    assert attrs["gen_ai.operation.name"] == "invoke_workflow"
    assert attrs["gen_ai.workflow.name"] == "ContentApprovalFlow"


def test_gen_ai_io_wraps_values_in_message_schema_and_drops_none() -> None:
    both = semantic_conventions.gen_ai_io(
        input_value='{"topic": "ai"}', output_value="final result"
    )
    assert json.loads(both["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": '{"topic": "ai"}'}]}
    ]
    assert json.loads(both["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "final result"}],
            "finish_reason": "stop",
        }
    ]

    assert semantic_conventions.gen_ai_io() == {}
    assert "gen_ai.output.messages" not in semantic_conventions.gen_ai_io(
        input_value="only-in"
    )


_VALUE_SHAPED_ATTRS = [
    ("to_system_instructions", "system_instructions", "gen_ai.system_instructions", "x"),
    ("to_tool_definitions", "tool_definitions", "gen_ai.tool.definitions", [{"name": "x"}]),
    ("to_tool_call_arguments", "tool_call_arguments", "gen_ai.tool.call.arguments", {"q": "x"}),
    ("to_tool_call_result", "tool_call_result", "gen_ai.tool.call.result", "ok"),
    ("to_input_messages", "input_messages", "gen_ai.input.messages", [{"role": "user"}]),
]


@pytest.mark.parametrize(
    ("transform_name", "kwarg", "attr_key", "value"),
    _VALUE_SHAPED_ATTRS,
)
def test_gen_ai_swallows_transform_failure_and_drops_attribute(
    transform_name: str,
    kwarg: str,
    attr_key: str,
    value: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=semantic_conventions.__name__)

    def boom(_v: Any) -> Any:
        raise RuntimeError("kaboom")

    with patch.object(semantic_conventions.gen_ai_shapes, transform_name, boom):
        attrs = semantic_conventions.gen_ai(**{kwarg: value})

    assert attr_key not in attrs
    assert any(attr_key in r.message for r in caplog.records)


def test_gen_ai_swallows_output_messages_failure_and_drops_finish_reasons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=semantic_conventions.__name__)

    def boom(_v: Any) -> Any:
        raise RuntimeError("kaboom")

    with patch.object(
        semantic_conventions.gen_ai_shapes, "to_output_messages", boom
    ):
        attrs = semantic_conventions.gen_ai(output_messages="anything")

    assert "gen_ai.output.messages" not in attrs
    assert "gen_ai.response.finish_reasons" not in attrs
    assert any("gen_ai.output.messages" in r.message for r in caplog.records)


class TestOutputTypeAndPayloadSize:
    def test_output_type_json_emitted_when_passed(self) -> None:
        attrs = semantic_conventions.gen_ai(operation_name="chat", output_type="json")

        assert attrs["gen_ai.output.type"] == "json"

    def test_output_type_text_emitted_when_passed(self) -> None:
        attrs = semantic_conventions.gen_ai(operation_name="chat", output_type="text")

        assert attrs["gen_ai.output.type"] == "text"

    def test_input_messages_size_matches_serialized_payload_length(self) -> None:
        attrs = semantic_conventions.gen_ai(
            input_messages=[{"role": "user", "content": "hi"}]
        )

        size = attrs["gen_ai.input.messages.size"]
        assert isinstance(size, int)
        assert size == len(attrs["gen_ai.input.messages"])
        assert size > 0

    def test_output_messages_size_matches_serialized_payload_length(self) -> None:
        attrs = semantic_conventions.gen_ai(output_messages="hello")

        size = attrs["gen_ai.output.messages.size"]
        assert isinstance(size, int)
        assert size == len(attrs["gen_ai.output.messages"])
        assert size > 0

    def test_size_and_output_type_attrs_absent_when_nothing_supplied(self) -> None:
        attrs = semantic_conventions.gen_ai(operation_name="chat")

        assert "gen_ai.output.type" not in attrs
        assert "gen_ai.input.messages.size" not in attrs
        assert "gen_ai.output.messages.size" not in attrs


def test_gen_ai_when_input_messages_exceed_cap_emits_truncation_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREWAI_OTEL_MAX_ATTR_BYTES", str(8 * 1024))
    huge = "A" * 50_000
    attrs = semantic_conventions.gen_ai(
        input_messages=[{"role": "user", "content": huge}],
        output_messages="ok",
    )

    assert attrs["gen_ai.input.messages.truncated"] is True
    assert attrs["gen_ai.input.messages.original_size_bytes"] > 50_000
    assert attrs["gen_ai.input.messages.size"] <= 8 * 1024
    assert "gen_ai.output.messages.truncated" not in attrs


def test_gen_ai_emits_sampling_params_and_explicit_finish_overrides_inference() -> None:
    # When the new OSS event fields are present, every sampling param lands
    # on its `gen_ai.request.*` attribute key, `response_id` lands on
    # `gen_ai.response.id`, and the explicit (Gemini-shaped) `finish_reason`
    # is coerced and overrides the value inferred from the output payload.
    attrs = semantic_conventions.gen_ai(
        operation_name="chat",
        output_messages="done",
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        stream=True,
        seed=42,
        stop_sequences=["END"],
        frequency_penalty=0.1,
        presence_penalty=0.2,
        choice_count=3,
        finish_reason="STOP",
        response_id="resp_123",
    )

    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["gen_ai.request.top_p"] == 0.95
    assert attrs["gen_ai.request.max_tokens"] == 1024
    assert attrs["gen_ai.request.stream"] is True
    assert attrs["gen_ai.request.seed"] == 42
    assert attrs["gen_ai.request.stop_sequences"] == ["END"]
    assert attrs["gen_ai.request.frequency_penalty"] == 0.1
    assert attrs["gen_ai.request.presence_penalty"] == 0.2
    assert attrs["gen_ai.request.choice.count"] == 3
    assert attrs["gen_ai.response.id"] == "resp_123"
    # Output payload would have inferred ["stop"] from a plain-string
    # response; the explicit Gemini-shaped finish_reason coerces to the
    # same OTel enum value and wins.
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


def test_gen_ai_explicit_finish_propagates_into_output_messages_payload() -> None:
    # Regression: previously the explicit finish_reason only overrode the
    # span-level `gen_ai.response.finish_reasons` attribute while the
    # embedded `gen_ai.output.messages[*].finish_reason` still carried the
    # inferred default ("stop"), giving a Gemini call with response="hi" +
    # finish_reason="MAX_TOKENS" inconsistent attributes (["length"] at the
    # span level, "stop" inside the message payload).
    attrs = semantic_conventions.gen_ai(
        operation_name="chat",
        output_messages="hi",
        finish_reason="MAX_TOKENS",
    )

    output_messages = json.loads(attrs["gen_ai.output.messages"])
    assert attrs["gen_ai.response.finish_reasons"] == ["length"]
    assert all(msg["finish_reason"] == "length" for msg in output_messages)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", "stop"),
        ("incomplete", "length"),
        ("failed", "error"),
        ("cancelled", "error"),
    ],
)
def test_gen_ai_maps_openai_responses_api_status_to_finish_reasons(
    status: str, expected: str
) -> None:
    attrs = semantic_conventions.gen_ai(
        operation_name="chat",
        output_messages="done",
        finish_reason=status,
    )

    assert attrs["gen_ai.response.finish_reasons"] == [expected]
    output_messages = json.loads(attrs["gen_ai.output.messages"])
    assert all(msg["finish_reason"] == expected for msg in output_messages)


def test_gen_ai_older_oss_path_omits_new_request_and_response_attrs() -> None:
    # Older crewai callers don't pass any of the new kwargs (their events
    # don't carry those fields). `_filter_none` must drop them entirely so
    # the emitted span doesn't acquire e.g. `gen_ai.response.id=None`.
    attrs = semantic_conventions.gen_ai(operation_name="chat")

    new_keys = {
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
        "gen_ai.request.max_tokens",
        "gen_ai.request.stream",
        "gen_ai.request.seed",
        "gen_ai.request.stop_sequences",
        "gen_ai.request.frequency_penalty",
        "gen_ai.request.presence_penalty",
        "gen_ai.request.choice.count",
        "gen_ai.response.id",
    }
    assert new_keys.isdisjoint(attrs.keys())
    # And without an explicit finish_reason or output payload, no inferred
    # finish reason is emitted either.
    assert "gen_ai.response.finish_reasons" not in attrs
