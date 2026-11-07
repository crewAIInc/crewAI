import json

from crewai.telemetry.tracing.gen_ai_shapes import to_input_messages, truncate_attr
import pytest


@pytest.mark.parametrize("cap", [1, 64, 128, 1024])
@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({"result": "result" * 1000}),
        "\x00" * 10000,
        "🤖" * 10000,
        json.dumps(
            [
                {
                    "role": "assistant",
                    "parts": [{"type": "tool_call", "arguments": "x" * 10000}],
                }
            ]
        ),
    ],
    ids=["object", "escaped", "unicode", "message-without-text"],
)
def test_truncation_preserves_json_and_respects_byte_cap(payload, cap):
    result, markers = truncate_attr(payload, attr="test.attribute", max_bytes=cap)

    assert markers == {
        "test.attribute.truncated": True,
        "test.attribute.original_size_bytes": len(payload.encode("utf-8")),
    }
    if cap < 128:
        assert result is None
    else:
        assert result is not None
        assert len(result.encode("utf-8")) <= cap
        assert json.loads(result)["_truncated"] is True


def test_multimodal_input_records_text_instead_of_provider_json():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        },
    ]

    assert to_input_messages(messages) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "Describe this image"}],
        },
    ]
