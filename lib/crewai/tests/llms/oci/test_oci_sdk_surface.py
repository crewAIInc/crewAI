from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crewai.llms.providers.oci.completion import OCICompletion
from crewai.llms.providers.oci.vision import (
    IMAGE_EMBEDDING_MODELS,
    VISION_MODELS,
    encode_image,
    is_vision_model,
    load_image,
    to_data_uri,
)


def test_oci_iter_stream_yields_text_chunks_and_metadata(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["stream"](
        {"message": {"content": [{"text": "Hello"}]}},
        {"message": {"content": [{"text": " world"}]}},
        {"finishReason": "stop"},
        {"usage": {"promptTokens": 3, "completionTokens": 2, "totalTokens": 5}},
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    chunks = list(
        llm.iter_stream([{"role": "user", "content": str(oci_unit_values["hello_prompt"])}])
    )

    assert chunks == ["Hello", " world"]
    assert llm.last_response_metadata == {
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


@pytest.mark.asyncio
async def test_oci_astream_yields_text_chunks(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.return_value = oci_response_factories["stream"](
        {"message": {"content": [{"text": "Async"}]}},
        {"message": {"content": [{"text": " stream"}]}},
        {"finishReason": "stop"},
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    chunks = []
    async for chunk in llm.astream(
        [{"role": "user", "content": str(oci_unit_values["hello_prompt"])}]
    ):
        chunks.append(chunk)

    assert chunks == ["Async", " stream"]


@pytest.mark.asyncio
async def test_oci_abatch_runs_multiple_calls(
    patch_oci_module, oci_response_factories, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.side_effect = [
        oci_response_factories["chat"]("first"),
        oci_response_factories["chat"]("second"),
    ]
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["generic_tool_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    results = await llm.abatch(["prompt one", "prompt two"])

    assert results == ["first", "second"]
    assert fake_client.chat.call_count == 2


def test_oci_extracts_response_metadata(
    patch_oci_module, oci_unit_values: dict[str, object]
):
    fake_client = MagicMock()
    fake_client.chat.return_value = SimpleNamespace(
        data=SimpleNamespace(
            chat_response=SimpleNamespace(
                finish_reason="stop",
                citations=[{"start": 0, "end": 1}],
                documents=[{"id": "doc_1"}],
                search_queries=["oracle cloud"],
                is_search_required=True,
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )
        )
    )
    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    llm = OCICompletion(
        model=str(oci_unit_values["cohere_model"]),
        compartment_id=str(oci_unit_values["compartment_id"]),
    )

    llm.call("hello")

    assert llm.last_response_metadata == {
        "finish_reason": "stop",
        "documents": [{"id": "doc_1"}],
        "citations": [{"start": 0, "end": 1}],
        "search_queries": ["oracle cloud"],
        "is_search_required": True,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_oci_vision_utilities(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert "meta.llama-3.2-90b-vision-instruct" in VISION_MODELS
    assert "cohere.embed-v4.0" in IMAGE_EMBEDDING_MODELS
    assert is_vision_model("google.gemini-2.5-flash") is True
    assert is_vision_model("meta.llama-3.3-70b-instruct") is False
    assert to_data_uri(b"hello", mime_type="image/png").startswith("data:image/png;base64,")
    assert to_data_uri("data:image/png;base64,abc") == "data:image/png;base64,abc"
    assert load_image(image_path)["image_url"]["url"].startswith("data:image/png;base64,")
    assert encode_image(b"hello", "image/png")["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
