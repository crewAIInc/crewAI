"""Native Ace Data Cloud routing, without paid requests."""

import json
from unittest.mock import patch

import httpx
import pytest
from crewai_core.acedatacloud import AceModel

from crewai import LLM
from crewai.llms.providers.openai_compatible import OpenAICompatibleCompletion


@pytest.fixture(autouse=True)
def published_catalog(monkeypatch):
    monkeypatch.setattr("crewai.llms.providers.acedatacloud.completion.get_ace_models", lambda: (
        AceModel("gpt-4.1-mini", "GPT-4.1 Mini", ("chat_completions",), True),
        AceModel("claude-sonnet-5-5", "Messages", ("messages",), True),
        AceModel("responses-only", "Responses", ("responses",), False),
    ))


def test_native_provider_defaults(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "test-consumer-key")
    monkeypatch.delenv("ACEDATACLOUD_BASE_URL", raising=False)
    llm = LLM(model="acedatacloud/gpt-4.1-mini")
    assert isinstance(llm, OpenAICompatibleCompletion)
    assert llm.provider == "acedatacloud"
    assert llm.model == "gpt-4.1-mini"
    assert str(llm._get_sync_client().base_url) == "https://api.acedata.cloud/v1/"
    assert llm._get_sync_client().api_key == "test-consumer-key"


def test_model_ids_with_slashes_are_preserved(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "test-consumer-key")
    llm = LLM(model="acedatacloud/public/model-id")
    assert llm.model == "public/model-id"


def test_explicit_provider_and_endpoint_override(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_BASE_URL", "https://gateway.example/v1")
    llm = LLM(model="current-model", provider="acedatacloud", api_key="explicit-key")
    assert llm.model == "current-model"
    assert str(llm._get_sync_client().base_url) == "https://gateway.example/v1/"
    assert llm._get_sync_client().api_key == "explicit-key"


def test_missing_key_fails_before_request(monkeypatch):
    monkeypatch.delenv("ACEDATACLOUD_API_KEY", raising=False)
    with pytest.raises((ValueError, ImportError), match="ACEDATACLOUD_API_KEY"):
        LLM(model="acedatacloud/gpt-4.1-mini")


def test_chat_completions_request_uses_native_provider(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "test-consumer-key")
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, headers={"content-type": "application/json"}, json={
            "id": "fixture", "object": "chat.completion", "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "fixture response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    with patch("httpx.Client.send", side_effect=respond):
        llm = LLM(model="acedatacloud/gpt-4.1-mini", max_retries=0)
        assert llm.call("test") == "fixture response"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.acedata.cloud/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-consumer-key"
    assert json.loads(requests[0].content)["model"] == "gpt-4.1-mini"


def test_messages_uses_bearer_and_does_not_inherit_vendor_key_or_limits(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-fixture-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-vendor-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unrelated.example")
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "native messages"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        })

    with patch("httpx.Client.send", side_effect=respond):
        llm = LLM(model="acedatacloud/claude-sonnet-5-5", max_retries=0)
        assert llm.call("Hello") == "native messages"
    request = requests[0]
    assert str(request.url) == "https://api.acedata.cloud/v1/messages"
    assert request.headers["authorization"] == "Bearer ace-fixture-key"
    assert "x-api-key" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 4096
    assert "thinking" not in body
    assert llm.get_context_window_size() == 4096
    assert llm.supports_multimodal() is True
    assert llm._supports_native_structured_output() is False


def test_responses_selection_and_explicit_context_override(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-fixture-key")
    llm = LLM(model="acedatacloud/responses-only", context_window_size=12000)
    assert llm.api == "responses"
    assert llm.get_context_window_size() == int(12000 * 0.85)


def test_custom_endpoint_does_not_query_official_metadata(monkeypatch):
    monkeypatch.setattr("crewai.llms.providers.acedatacloud.completion.get_ace_models", lambda: pytest.fail("custom endpoint read official metadata"))
    llm = LLM(model="acedatacloud/custom/alias", api_key="custom-key", base_url="https://custom.example/v1")
    assert llm.model == "custom/alias"
    assert str(llm._get_sync_client().base_url) == "https://custom.example/v1/"


def test_native_messages_stream_preserves_text_and_usage(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-fixture-key")
    events = [
        {"type": "message_start", "message": {"id": "msg_stream", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [], "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "native stream"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=body)

    with patch("httpx.Client.send", side_effect=respond):
        llm = LLM(model="acedatacloud/claude-sonnet-5-5", stream=True, max_retries=0)
        assert llm.call("Hello") == "native stream"
    assert str(requests[0].url) == "https://api.acedata.cloud/v1/messages"
    assert json.loads(requests[0].content)["stream"] is True
    assert llm.get_token_usage_summary().prompt_tokens == 2
    assert llm.get_token_usage_summary().completion_tokens == 3


def test_native_messages_errors_do_not_fall_back_to_chat(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-fixture-key")
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return httpx.Response(403, request=request, json={"error": {"type": "permission_error", "message": "Access denied"}})

    with patch("httpx.Client.send", side_effect=respond), pytest.raises(Exception, match="Access denied"):
        LLM(model="acedatacloud/claude-sonnet-5-5", max_retries=0).call("Hello")
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.acedata.cloud/v1/messages"


def test_native_messages_returns_tool_calls_for_the_executor(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-fixture-key")
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "msg_tool", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"value": "sample"}}],
            "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 3},
        })

    with patch("httpx.Client.send", side_effect=respond):
        llm = LLM(model="acedatacloud/claude-sonnet-5-5", max_retries=0)
        result = llm.call("Look it up", tools=[{"type": "function", "function": {"name": "lookup", "description": "Look up a value", "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}}])
    assert result[0].name == "lookup" and result[0].input == {"value": "sample"}
    body = json.loads(requests[0].content)
    assert body["tools"][0]["name"] == "lookup"
    assert body["tools"][0]["input_schema"]["properties"]["value"]["type"] == "string"


@pytest.mark.asyncio
async def test_async_messages_uses_the_same_protocol_and_bearer(monkeypatch):
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-fixture-key")
    requests = []

    async def respond(request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "msg_async", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "native async"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        })

    with patch("httpx.AsyncClient.send", side_effect=respond):
        llm = LLM(model="acedatacloud/claude-sonnet-5-5", max_retries=0)
        assert await llm.acall("Hello") == "native async"
    assert str(requests[0].url) == "https://api.acedata.cloud/v1/messages"
    assert requests[0].headers["authorization"] == "Bearer ace-fixture-key"
    assert "x-api-key" not in requests[0].headers
