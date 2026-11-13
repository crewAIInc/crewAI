"""Azure streaming keeps cached and reasoning token counts in usage.

The non-streaming path reads ``prompt_tokens_details.cached_tokens`` and
``completion_tokens_details.reasoning_tokens``; the streaming paths built the
usage dict from the three totals only, so both counts were always 0.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.ai.inference.models import StreamingChatCompletionsUpdate

from crewai.llms.providers.azure.completion import AzureCompletion


def _chunks():
    content = StreamingChatCompletionsUpdate(
        id="chatcmpl-1",
        model="gpt-4o",
        created=None,
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
    )
    usage = StreamingChatCompletionsUpdate(
        id="chatcmpl-1",
        model="gpt-4o",
        created=None,
        choices=[],
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    )
    return [content, usage]


def _llm():
    return AzureCompletion(
        model="gpt-4o",
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
        stream=True,
    )


def _assert_usage(llm):
    summary = llm.get_token_usage_summary()
    assert summary.prompt_tokens == 100
    assert summary.completion_tokens == 200
    assert summary.total_tokens == 300
    assert summary.cached_prompt_tokens == 40
    assert summary.reasoning_tokens == 60


def test_streaming_usage_keeps_cached_and_reasoning_tokens():
    llm = _llm()
    with patch.object(llm._client, "complete", return_value=iter(_chunks())):
        result = llm.call([{"role": "user", "content": "Say hello"}])

    assert result == "Hello"
    _assert_usage(llm)


@pytest.mark.asyncio
async def test_async_streaming_usage_keeps_cached_and_reasoning_tokens():
    async def stream():
        for chunk in _chunks():
            yield chunk

    llm = _llm()
    client = MagicMock()
    client.complete = AsyncMock(return_value=stream())
    with patch.object(llm, "_get_async_client", return_value=client):
        result = await llm.acall([{"role": "user", "content": "Say hello"}])

    assert result == "Hello"
    _assert_usage(llm)


def test_non_streaming_usage_reads_details_from_the_sdk_model():
    from azure.ai.inference.models import ChatCompletions

    response = ChatCompletions(
        id="chatcmpl-1",
        model="gpt-4o",
        created=0,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    )

    usage = AzureCompletion._extract_azure_token_usage(response)

    assert usage["cached_prompt_tokens"] == 40
    assert usage["reasoning_tokens"] == 60
