"""Native Ace Data Cloud routing, without paid requests."""

import json
from unittest.mock import patch

import httpx
import pytest

from crewai import LLM
from crewai.llms.providers.openai_compatible import OpenAICompatibleCompletion


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
