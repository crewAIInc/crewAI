"""Tests for crewAIInc/crewAI#8009 — native Groq provider routing.

Issue: ``LLM(model="groq/<model-id>")`` should use CrewAI's native
``OpenAICompatibleCompletion`` instead of the LiteLLM fallback.
Spec from the issue: default endpoint https://api.groq.com/openai/v1,
API key env GROQ_API_KEY, endpoint override following the existing
``<PROVIDER>_BASE_URL`` convention.

Baseline (crewai 1.15.40 from PyPI): every test below fails — groq is absent
from all routing tables and the completion class rejects the provider.
Patched: all pass. No network access is used anywhere in these tests.
"""

import pytest

# This sandbox exports proxy variables whose no_proxy list contains
# bracketed IPv6 entries that httpx cannot parse when a client is built.
# Clear them so client construction is hermetic in any environment.
_PROXY_VARS = [
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
]


@pytest.fixture(autouse=True)
def _no_proxy_env(monkeypatch):
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)


from crewai import LLM
from crewai.llm import PROVIDER_ALIASES, SUPPORTED_NATIVE_PROVIDERS
from crewai.llms.base_llm import MissingAPIKeyError
from crewai.llms.providers.openai_compatible.completion import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OpenAICompatibleCompletion,
)

MODEL_ID = "llama-3.1-8b-instant"


def test_groq_registered_in_provider_registry():
    cfg = OPENAI_COMPATIBLE_PROVIDERS["groq"]
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.api_key_env == "GROQ_API_KEY"
    assert cfg.base_url_env == "GROQ_BASE_URL"
    assert cfg.api_key_required is True


def test_groq_in_native_provider_tables():
    assert "groq" in SUPPORTED_NATIVE_PROVIDERS
    assert PROVIDER_ALIASES["groq"] == "groq"


def test_factory_routes_groq_to_native_completion():
    assert LLM._get_native_provider("groq") is OpenAICompatibleCompletion


def test_completion_resolves_default_endpoint(monkeypatch):
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    c = OpenAICompatibleCompletion(
        model=MODEL_ID, provider="groq", api_key="test-key"
    )
    assert c.base_url == "https://api.groq.com/openai/v1"


def test_completion_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key-123")
    c = OpenAICompatibleCompletion(model=MODEL_ID, provider="groq")
    assert c.api_key == "env-key-123"


def test_completion_base_url_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_BASE_URL", "https://groq-proxy.example.com/v1")
    c = OpenAICompatibleCompletion(
        model=MODEL_ID, provider="groq", api_key="test-key"
    )
    assert c.base_url == "https://groq-proxy.example.com/v1"


def test_missing_key_raises_on_client_params(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    c = OpenAICompatibleCompletion(model=MODEL_ID, provider="groq")
    with pytest.raises(MissingAPIKeyError):
        c._get_client_params()


def test_model_string_routes_native_end_to_end():
    route = LLM._resolve_route(f"groq/{MODEL_ID}", {})
    assert route.provider == "groq"
    assert route.native_class is OpenAICompatibleCompletion
    # Full provider model ID is preserved after the prefix.
    assert route.model.endswith(MODEL_ID)
