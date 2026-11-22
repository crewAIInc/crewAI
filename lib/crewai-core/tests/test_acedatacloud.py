"""The full public model contract determines identity, modality and protocol."""

from unittest.mock import patch

import httpx
import pytest

from crewai_core.acedatacloud import API_BASE, CATALOG_URL, _catalog, get_ace_models


def row(model, *, modality="chat", protocol="chat_completions"):
    path = {"chat_completions": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}[protocol]
    return {"id": model, "modality": modality, "capabilities": ["vision"], "endpoints": [{"protocol": protocol, "path": path, "base_url": "https://api.acedata.cloud" if protocol == "messages" else API_BASE, "auth": "bearer"}]}


@pytest.fixture(autouse=True)
def clear_catalog():
    _catalog.cache_clear()
    yield
    _catalog.cache_clear()


def test_catalog_retains_all_typed_chat_ids_without_key_or_name_guesses():
    ids = [f"model-{index}" for index in range(20)] + ["gpt-4o-image", "messages-only"]
    rows = [row(model) for model in ids[:-1]] + [row("messages-only", protocol="messages"), row("chat-sounding", modality="image")]
    response = httpx.Response(200, request=httpx.Request("GET", CATALOG_URL), json={"items": rows})
    with patch("crewai_core.acedatacloud.httpx.get", return_value=response) as get:
        models = get_ace_models()
        assert [model.id for model in models] == ids
        assert models[-1].protocols == ("messages",)
        assert models[-1].vision is True
        assert get_ace_models() == models
    get.assert_called_once_with(CATALOG_URL, timeout=6, follow_redirects=False)


def test_missing_or_untrusted_protocol_metadata_does_not_guess_openai():
    item = row("example", protocol="messages")
    item["endpoints"][0]["base_url"] = "https://not-ace.example"
    response = httpx.Response(200, request=httpx.Request("GET", CATALOG_URL), json={"items": [item]})
    with patch("crewai_core.acedatacloud.httpx.get", return_value=response), pytest.raises(ValueError, match="No supported"):
        get_ace_models()
