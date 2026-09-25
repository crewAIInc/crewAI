"""Structured output must collapse multimodal content with the shared helper."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from crewai.llm import LLM


class Answer(BaseModel):
    text: str


MULTIMODAL = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
        ],
    }
]


def _stub_llm():
    """An LLM with the instructor call captured, no network involved."""
    llm = object.__new__(LLM)
    llm.is_litellm = True
    llm._handle_emit_call_events = lambda **kwargs: None
    captured: dict[str, str] = {}

    class FakeInstructor:
        def __init__(self, *, content, model, llm):
            captured["content"] = content

        def to_pydantic(self):
            return Answer(text="ok")

    patcher = patch(
        "crewai.utilities.internal_instructor.InternalInstructor", FakeInstructor
    )
    return llm, captured, patcher


def test_sync_structured_output_collapses_multimodal_content():
    llm, captured, patcher = _stub_llm()
    with patcher:
        llm._handle_non_streaming_response(
            params={"messages": MULTIMODAL}, response_model=Answer
        )
    assert captured["content"] == "USER: What is in this image?"


@pytest.mark.asyncio
async def test_async_structured_output_collapses_multimodal_content():
    llm, captured, patcher = _stub_llm()
    with patcher:
        await llm._ahandle_non_streaming_response(
            params={"messages": MULTIMODAL}, response_model=Answer
        )
    assert captured["content"] == "USER: What is in this image?"
