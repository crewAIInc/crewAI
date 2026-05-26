"""Unit tests for OCI provider streaming (mocked SDK)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_fake_stream_event(text: str = "", finish_reason: str | None = None, usage: dict | None = None) -> MagicMock:
    """Build a single SSE event with optional text, finish, and usage."""
    payload: dict = {}
    if text:
        payload["message"] = {"content": [{"text": text}]}
    if finish_reason:
        payload["finishReason"] = finish_reason
    if usage:
        payload["usage"] = usage

    import json
    event = MagicMock()
    event.data = json.dumps(payload)
    return event


def _make_fake_stream_response(*events: MagicMock) -> MagicMock:
    """Wrap events into a response.data.events() iterable."""
    response = MagicMock()
    response.data.events.return_value = iter(events)
    return response


def test_oci_completion_streams_generic_responses(
    patch_oci_module, oci_unit_values
):
    """Streaming call should accumulate text chunks and return full response."""
    from crewai.llms.providers.oci.completion import OCICompletion

    events = [
        _make_fake_stream_event(text="Hello "),
        _make_fake_stream_event(text="world"),
        _make_fake_stream_event(
            finish_reason="stop",
            usage={"promptTokens": 5, "completionTokens": 2, "totalTokens": 7},
        ),
    ]
    fake_client = MagicMock()
    fake_client.chat.return_value = _make_fake_stream_response(*events)
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client
    # StreamOptions mock
    patch_oci_module.generative_ai_inference.models.StreamOptions = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
        stream=True,
    )
    result = llm.call(messages=[{"role": "user", "content": "Say hello"}])

    assert "Hello " in result
    assert "world" in result
    assert llm.last_response_metadata is not None
    assert llm.last_response_metadata.get("finish_reason") == "stop"


def test_oci_iter_stream_yields_text_chunks(
    patch_oci_module, oci_unit_values
):
    """iter_stream should yield individual text chunks."""
    from crewai.llms.providers.oci.completion import OCICompletion

    events = [
        _make_fake_stream_event(text="chunk1"),
        _make_fake_stream_event(text="chunk2"),
        _make_fake_stream_event(
            usage={"promptTokens": 3, "completionTokens": 2, "totalTokens": 5},
        ),
    ]
    fake_client = MagicMock()
    fake_client.chat.return_value = _make_fake_stream_response(*events)
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client
    patch_oci_module.generative_ai_inference.models.StreamOptions = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    chunks = list(llm.iter_stream(messages=[{"role": "user", "content": "test"}]))

    assert chunks == ["chunk1", "chunk2"]
    assert llm.last_response_metadata is not None
    assert llm.last_response_metadata["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_oci_astream_yields_text_chunks(
    patch_oci_module, oci_unit_values
):
    """astream should yield chunks via async generator."""
    from crewai.llms.providers.oci.completion import OCICompletion

    events = [
        _make_fake_stream_event(text="async1"),
        _make_fake_stream_event(text="async2"),
        _make_fake_stream_event(),
    ]
    fake_client = MagicMock()
    fake_client.chat.return_value = _make_fake_stream_response(*events)
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client
    patch_oci_module.generative_ai_inference.models.StreamOptions = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    chunks = []
    async for chunk in llm.astream(messages=[{"role": "user", "content": "test"}]):
        chunks.append(chunk)

    assert chunks == ["async1", "async2"]


def test_oci_stream_chat_events_holds_client_lock(
    patch_oci_module, oci_unit_values
):
    """_stream_chat_events should hold the client lock for the full iteration."""
    from crewai.llms.providers.oci.completion import OCICompletion

    events = [_make_fake_stream_event(text="a"), _make_fake_stream_event(text="b")]
    fake_client = MagicMock()
    fake_client.chat.return_value = _make_fake_stream_response(*events)
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = fake_client

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )

    # Before streaming, ticket should be 0
    assert llm._active_client_ticket == 0
    chat_details = MagicMock()
    list(llm._stream_chat_events(chat_details))
    # After streaming completes, ticket should have advanced
    assert llm._active_client_ticket == 1
