"""Unit tests for OCI Cohere v2 (COHEREV2) api-format support (mocked SDK)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_v2_llm(patch_oci_module, oci_unit_values, **kwargs):
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    return OCICompletion(
        model="cohere.command-a-vision",
        compartment_id=oci_unit_values["compartment_id"],
        **kwargs,
    )


def test_oci_infers_cohere_v2_for_command_a_vision(patch_oci_module, oci_unit_values):
    """command-a-vision must route to the COHEREV2 api format, not v1."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)
    assert llm.oci_provider == "cohere_v2"


def test_oci_infers_cohere_v1_for_command_a(patch_oci_module, oci_unit_values):
    """Plain command-a keeps using the v1 COHERE api format."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model="cohere.command-a-03-2025",
        compartment_id=oci_unit_values["compartment_id"],
    )
    assert llm.oci_provider == "cohere"


def test_oci_cohere_v2_chat_request_uses_v2_format(patch_oci_module, oci_unit_values):
    """The request must be a CohereChatRequestV2 with api_format COHEREV2 and v2 messages."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values, max_tokens=64)
    llm._build_chat_request([{"role": "user", "content": "hello"}])

    request_cls = patch_oci_module.generative_ai_inference.models.CohereChatRequestV2
    request_cls.assert_called_once()
    kwargs = request_cls.call_args.kwargs
    assert kwargs["api_format"] == "COHEREV2"
    assert kwargs["max_tokens"] == 64
    patch_oci_module.generative_ai_inference.models.CohereUserMessageV2.assert_called_once()


def test_oci_cohere_v2_builds_image_content(patch_oci_module, oci_unit_values):
    """image_url content should produce CohereImageContentV2 with CohereImageUrlV2."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)
    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
    ]
    result = llm._build_cohere_v2_content(content)

    assert len(result) == 2
    patch_oci_module.generative_ai_inference.models.CohereImageContentV2.assert_called_once()
    patch_oci_module.generative_ai_inference.models.CohereImageUrlV2.assert_called_once()


def test_oci_cohere_v2_rejects_unknown_content_type(patch_oci_module, oci_unit_values):
    """Unsupported content types should raise instead of being silently dropped."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)
    with pytest.raises(ValueError, match="text and image_url"):
        llm._build_cohere_v2_content([{"type": "video_url", "video_url": {"url": "x"}}])


def test_oci_cohere_v2_formats_tools(patch_oci_module, oci_unit_values):
    """Tools should be CohereToolV2(type=FUNCTION, function=Function(...))."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    formatted = llm._format_tools(tools)

    assert len(formatted) == 1
    tool_cls = patch_oci_module.generative_ai_inference.models.CohereToolV2
    tool_cls.assert_called_once()
    assert tool_cls.call_args.kwargs["type"] == "FUNCTION"
    fn_cls = patch_oci_module.generative_ai_inference.models.Function
    assert fn_cls.call_args.kwargs["name"] == "get_weather"


def test_oci_cohere_v2_extracts_message_text(patch_oci_module, oci_unit_values):
    """Response text lives on chat_response.message.content[].text."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)

    part = MagicMock()
    part.text = "v2 says hi"
    message = MagicMock()
    message.content = [part]
    chat_response = MagicMock()
    chat_response.text = None
    chat_response.message = message
    response = MagicMock()
    response.data.chat_response = chat_response

    assert llm._extract_text(response) == "v2 says hi"


def test_oci_cohere_v2_extracts_tool_calls(patch_oci_module, oci_unit_values):
    """Tool calls come from message.tool_calls with a nested function object."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)

    function = MagicMock()
    function.name = "get_weather"
    function.arguments = '{"city": "Toronto"}'
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function = function
    message = MagicMock()
    message.tool_calls = [tool_call]
    chat_response = MagicMock()
    chat_response.message = message
    response = MagicMock()
    response.data.chat_response = chat_response

    calls = llm._extract_tool_calls(response)
    assert calls == [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Toronto"}'},
    }]


def test_oci_cohere_v2_stream_event_text(patch_oci_module, oci_unit_values):
    """V2 SSE payloads carry text under message.content[].text (verified live)."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)
    event = {
        "apiFormat": "COHEREV2",
        "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "chunk"}]},
    }
    assert llm._extract_text_from_stream_event(event) == "chunk"


def test_oci_cohere_v2_stream_event_tool_calls(patch_oci_module, oci_unit_values):
    """V2 stream tool calls nest name/arguments under function."""
    llm = _make_v2_llm(patch_oci_module, oci_unit_values)
    event = {
        "message": {
            "toolCalls": [{
                "id": "c1",
                "type": "FUNCTION",
                "function": {"name": "f", "arguments": "{}"},
            }]
        }
    }
    calls = llm._extract_tool_calls_from_stream_event(event)
    assert calls == [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
