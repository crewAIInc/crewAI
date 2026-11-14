"""Regression tests for token usage riding on Azure streaming content chunks.

OpenAI-compatible servers can attach ``usage`` to chunks that still carry a
delta (vLLM does on every chunk when the request sets
``continuous_usage_stats``). The streaming handlers must record the usage
without dropping the chunk's content, in both the sync and async paths.
"""

from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.inference.models import StreamingChatCompletionsUpdate

from crewai.llms.providers.azure.completion import AzureCompletion


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _content_chunk(content: str, usage: dict | None = None) -> StreamingChatCompletionsUpdate:
    return StreamingChatCompletionsUpdate(
        id="chatcmpl-1",
        model="gpt-4o",
        created=None,
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }
        ],
        usage=usage,
    )


def _usage_only_chunk(usage: dict) -> StreamingChatCompletionsUpdate:
    return StreamingChatCompletionsUpdate(
        id="chatcmpl-1",
        model="gpt-4o",
        created=None,
        choices=[],
        usage=usage,
    )


def _stream_with_usage_on_content_chunk() -> list[StreamingChatCompletionsUpdate]:
    return [
        _content_chunk("Hello"),
        _content_chunk(" world", usage=_usage(10, 2)),
        _usage_only_chunk(_usage(10, 5)),
    ]


def _llm() -> AzureCompletion:
    return AzureCompletion(
        model="gpt-4o",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
        stream=True,
    )


def test_azure_streaming_keeps_delta_when_chunk_also_carries_usage():
    """A chunk with both content and usage must contribute its delta."""
    llm = _llm()

    with patch.object(llm._client, "complete") as mock_complete:
        mock_complete.return_value = iter(_stream_with_usage_on_content_chunk())

        result = llm.call([{"role": "user", "content": "Say hi"}])

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_azure_async_streaming_keeps_delta_when_chunk_also_carries_usage():
    """The async streaming path must keep the delta too."""
    llm = _llm()

    async def _stream() -> StreamingChatCompletionsUpdate:
        for chunk in _stream_with_usage_on_content_chunk():
            yield chunk

    mock_complete = AsyncMock(return_value=_stream())
    with patch.object(llm._async_client, "complete", mock_complete):
        result = await llm.acall([{"role": "user", "content": "Say hi"}])

    assert result == "Hello world"


def test_azure_streaming_still_records_usage_from_content_chunk():
    """Usage attached to a content chunk is still tracked."""
    llm = _llm()

    with patch.object(llm._client, "complete") as mock_complete:
        mock_complete.return_value = iter(_stream_with_usage_on_content_chunk())

        llm.call([{"role": "user", "content": "Say hi"}])

    metrics = llm.get_token_usage_summary()
    assert metrics.total_tokens == 15
    assert metrics.prompt_tokens == 10
    assert metrics.completion_tokens == 5
