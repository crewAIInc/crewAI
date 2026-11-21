"""Offline request-contract regressions for OpenAI-style Bedrock content."""

from copy import deepcopy
from typing import Any, Literal

from botocore.stub import Stubber
from crewai import LLM
from crewai.llms.providers.bedrock import completion as bedrock_module
from crewai.llms.providers.bedrock.completion import BedrockCompletion
from crewai.utilities.types import LLMMessage
import pytest


@pytest.fixture
def bedrock(monkeypatch: pytest.MonkeyPatch) -> BedrockCompletion:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert isinstance(llm, BedrockCompletion)
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([{"type": "text", "text": "look"}], [{"text": "look"}]),
        (
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw=="},
                }
            ],
            [{"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}],
        ),
        (
            [{"type": "image_url", "image_url": {"url": "data:image/jpg;base64,/9g="}}],
            [{"image": {"format": "jpeg", "source": {"bytes": b"\xff\xd8"}}}],
        ),
        (
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9g="},
                }
            ],
            [{"image": {"format": "jpeg", "source": {"bytes": b"\xff\xd8"}}}],
        ),
        (
            [{"text": "native"}, {"type": "text", "text": "converted"}],
            [{"text": "native"}, {"text": "converted"}],
        ),
        (
            [{"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}],
            [{"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}],
        ),
    ],
    ids=["text", "base64-image", "jpg-alias", "jpeg", "mixed", "native-image"],
)
async def test_public_call_sends_valid_converse_content(
    bedrock: BedrockCompletion,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    role: Literal["user", "assistant"],
    content: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> None:
    # The SDK validates the real request before returning the offline response.
    messages: list[LLMMessage] = [
        {"role": "user", "content": "start"},
        {"role": role, "content": content},
        {"role": "user", "content": "finish"},
    ]
    original = deepcopy(messages)
    expected_messages = [
        {"role": "user", "content": [{"text": "start"}]},
        {"role": role, "content": expected},
        {"role": "user", "content": [{"text": "finish"}]},
    ]
    with Stubber(bedrock._client) as stub:
        stub.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "ok"}]}
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
                "metrics": {"latencyMs": 1},
            },
            expected_params={
                "modelId": bedrock.model,
                "messages": expected_messages,
                "inferenceConfig": bedrock._get_inference_config(),
            },
        )
        if mode == "async":
            # Keep the provider's native async request path, substituting only transport.
            class OfflineAsyncClient:
                async def converse(self, **kwargs: Any) -> Any:
                    return bedrock._client.converse(**kwargs)

            monkeypatch.setattr(bedrock_module, "AIOBOTOCORE_AVAILABLE", True)
            bedrock._async_client_initialized = True
            bedrock._async_client = OfflineAsyncClient()
            result = await bedrock.acall(messages)
        else:
            result = bedrock.call(messages)
        assert result == "ok"
        stub.assert_no_pending_responses()
    assert messages == original
