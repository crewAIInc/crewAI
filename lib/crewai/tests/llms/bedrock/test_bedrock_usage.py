from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

from crewai import LLM
from crewai.flow.flow import Flow, start
from crewai.llms.base_llm import BaseLLM
from crewai.types.usage_metrics import UsageMetrics
import pytest


@pytest.fixture
def bedrock_llm(monkeypatch: pytest.MonkeyPatch) -> BaseLLM:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return LLM(model="bedrock/anthropic.claude-sonnet-4-5-v1:0")


@pytest.mark.parametrize(
    ("cache_usage", "expected_prompt", "expected_read", "expected_write"),
    [
        ({"cacheReadInputTokens": 1000, "cacheWriteInputTokens": 50}, 1250, 1000, 50),
        (
            {"cacheReadInputTokenCount": 1000, "cacheWriteInputTokenCount": 50},
            1250,
            1000,
            50,
        ),
        (
            {
                "cacheReadInputTokenCount": 1000,
                "cacheReadInputTokens": 1000,
                "cacheWriteInputTokenCount": 50,
                "cacheWriteInputTokens": 50,
            },
            1250,
            1000,
            50,
        ),
        (
            {
                "cacheReadInputTokenCount": 0,
                "cacheReadInputTokens": 1000,
                "cacheWriteInputTokenCount": 0,
                "cacheWriteInputTokens": 50,
            },
            1250,
            1000,
            50,
        ),
        ({"cacheReadInputTokens": 1000}, 1200, 1000, 0),
        ({"cacheWriteInputTokens": 50}, 250, 0, 50),
        ({}, 200, 0, 0),
    ],
    ids=[
        "cache-read-and-write",
        "legacy-cache-keys",
        "duplicate-aliases",
        "zero-legacy-aliases",
        "cache-read-only",
        "cache-write-only",
        "uncached",
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["call", "acall"])
@pytest.mark.asyncio
async def test_bedrock_cache_usage(
    bedrock_llm: BaseLLM,
    cache_usage: dict[str, int],
    expected_prompt: int,
    expected_read: int,
    expected_write: int,
    asynchronous: bool,
) -> None:
    usage = {
        "inputTokens": 200,
        "outputTokens": 100,
        "totalTokens": 300,
        **cache_usage,
    }
    original_usage = deepcopy(usage)
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": usage,
    }
    expected = UsageMetrics(
        prompt_tokens=expected_prompt,
        completion_tokens=100,
        total_tokens=expected_prompt + 100,
        cached_prompt_tokens=expected_read,
        cache_creation_tokens=expected_write,
        successful_requests=1,
    )

    for request_count in (1, 2):
        if asynchronous:
            client = AsyncMock()
            client.converse.return_value = response
            with patch.object(
                bedrock_llm, "_ensure_async_client", AsyncMock(return_value=client)
            ):
                result = await bedrock_llm.acall("hi")
        else:
            client = MagicMock()
            client.converse.return_value = response
            with patch.object(bedrock_llm, "_get_sync_client", return_value=client):
                result = bedrock_llm.call("hi")

        assert result == "hi"
        assert bedrock_llm.get_token_usage_summary().model_dump() == {
            key: value * request_count for key, value in expected.model_dump().items()
        }

    assert UsageMetrics.from_provider_dict(usage) == expected
    assert UsageMetrics.from_provider_dict(expected.model_dump()) == expected
    assert usage == original_usage


def test_flow_and_bedrock_report_the_same_cache_usage(bedrock_llm: BaseLLM) -> None:
    class UsageFlow(Flow):
        @start()
        def query(self) -> str:
            return str(bedrock_llm.call("hi"))

    client = MagicMock()
    client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 200,
            "outputTokens": 100,
            "totalTokens": 300,
            "cacheReadInputTokens": 1000,
            "cacheWriteInputTokens": 50,
        },
    }
    flow = UsageFlow()
    with patch.object(bedrock_llm, "_get_sync_client", return_value=client):
        assert flow.kickoff() == "hi"

    assert flow.usage_metrics == UsageMetrics(
        prompt_tokens=1250,
        completion_tokens=100,
        total_tokens=1350,
        cached_prompt_tokens=1000,
        cache_creation_tokens=50,
        successful_requests=1,
    )
    assert flow.usage_metrics == bedrock_llm.get_token_usage_summary()
