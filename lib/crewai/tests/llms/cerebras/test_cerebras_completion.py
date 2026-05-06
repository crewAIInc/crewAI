"""Tests for the native Cerebras provider.

Two flavors:
- Unit tests under the ``Test*`` classes — exercise factory routing, field
  normalization, env-var resolution, client construction, and the fallback
  to OpenAI-compatible when the SDK extra is not installed. These run with
  ``CEREBRAS_API_KEY`` / ``CEREBRAS_BASE_URL`` cleared so each test states
  the env it depends on.
- Module-level VCR tests — replay real Cerebras API responses from cassettes.
  To re-record, set ``CEREBRAS_API_KEY`` and run with
  ``PYTEST_VCR_RECORD_MODE=new_episodes`` (or delete the target cassette
  and use the default ``once`` mode).
"""

from __future__ import annotations

import builtins
import sys
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.cerebras.completion import CerebrasCompletion
from crewai.llms.providers.openai_compatible.completion import (
    OpenAICompatibleCompletion,
)


@pytest.fixture
def clear_cerebras_env(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_BASE_URL", raising=False)


@pytest.mark.usefixtures("clear_cerebras_env")
class TestCerebrasFactoryRouting:
    def test_provider_prefix_routes_to_native(self):
        llm = LLM(model="cerebras/gpt-oss-120b", api_key="sk-test")
        assert isinstance(llm, CerebrasCompletion)
        assert llm.llm_type == "cerebras"
        assert llm.provider == "cerebras"
        assert llm.is_litellm is False

    def test_explicit_provider_kwarg_routes_to_native(self):
        llm = LLM(model="gpt-oss-120b", provider="cerebras", api_key="sk-test")
        assert isinstance(llm, CerebrasCompletion)

    def test_falls_back_to_openai_compat_when_sdk_missing(self, monkeypatch):
        # Drop any cached imports so the factory re-imports the cerebras module.
        for mod_name in list(sys.modules):
            if mod_name.startswith(
                "crewai.llms.providers.cerebras"
            ) or mod_name.startswith("cerebras"):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)

        real_import = builtins.__import__

        def _import_blocker(name, *args, **kwargs):
            if name.startswith("cerebras"):
                raise ImportError(f"simulated missing dep: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import_blocker)

        llm = LLM(model="cerebras/gpt-oss-120b", api_key="sk-test")
        # Without the SDK, factory falls through to the OpenAI-compatible path.
        assert isinstance(llm, OpenAICompatibleCompletion)


@pytest.mark.usefixtures("clear_cerebras_env")
class TestCerebrasFieldNormalization:
    def test_default_base_url_left_unset(self):
        # We deliberately don't pin a default base URL — the SDK has its own.
        llm = CerebrasCompletion(model="gpt-oss-120b", api_key="sk-test")
        assert llm.base_url is None
        assert llm.api == "completions"
        assert llm.provider == "cerebras"

    def test_env_var_base_url_override(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_BASE_URL", "https://custom.cerebras.example/v1")
        llm = CerebrasCompletion(model="gpt-oss-120b", api_key="sk-test")
        assert llm.base_url == "https://custom.cerebras.example/v1"

    def test_explicit_base_url_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_BASE_URL", "https://from-env.example/v1")
        llm = CerebrasCompletion(
            model="gpt-oss-120b",
            api_key="sk-test",
            base_url="https://explicit.example/v1",
        )
        assert llm.base_url == "https://explicit.example/v1"

    def test_api_forced_to_completions(self):
        # Even if a caller tries to set api="responses", the validator clamps it.
        llm = CerebrasCompletion(
            model="gpt-oss-120b", api_key="sk-test", api="responses"
        )
        assert llm.api == "completions"


@pytest.mark.usefixtures("clear_cerebras_env")
class TestCerebrasApiKeyResolution:
    def test_env_var_picked_up(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "env-key")
        llm = CerebrasCompletion(model="gpt-oss-120b")
        assert llm.api_key == "env-key"

    def test_explicit_api_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "env-key")
        llm = CerebrasCompletion(model="gpt-oss-120b", api_key="explicit-key")
        assert llm.api_key == "explicit-key"

    def test_construction_succeeds_without_key(self):
        # Lazy client init: missing key should not crash construction.
        llm = CerebrasCompletion(model="gpt-oss-120b")
        assert llm.api_key is None

    def test_get_client_params_raises_without_key(self):
        llm = CerebrasCompletion(model="gpt-oss-120b")
        with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
            llm._get_client_params()


@pytest.mark.usefixtures("clear_cerebras_env")
class TestCerebrasClientBuild:
    def test_sync_client_uses_cerebras_sdk(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
        llm = CerebrasCompletion(model="gpt-oss-120b")
        client = llm._get_sync_client()
        from cerebras.cloud.sdk import Cerebras

        assert isinstance(client, Cerebras)

    def test_async_client_uses_cerebras_sdk(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
        llm = CerebrasCompletion(model="gpt-oss-120b")
        client = llm._get_async_client()
        from cerebras.cloud.sdk import AsyncCerebras

        assert isinstance(client, AsyncCerebras)

    def test_client_params_threaded_through(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
        llm = CerebrasCompletion(
            model="gpt-oss-120b",
            timeout=30.0,
            max_retries=5,
            default_headers={"X-Custom": "yes"},
        )
        params = llm._get_client_params()
        assert params["timeout"] == 30.0
        assert params["max_retries"] == 5
        assert params["default_headers"] == {"X-Custom": "yes"}
        assert params["api_key"] == "sk-test"
        # base_url omitted so the SDK uses its own default.
        assert "base_url" not in params

    def test_client_params_includes_base_url_when_set(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "sk-test")
        llm = CerebrasCompletion(
            model="gpt-oss-120b", base_url="https://override.example/api"
        )
        params = llm._get_client_params()
        assert params["base_url"] == "https://override.example/api"


@pytest.mark.usefixtures("clear_cerebras_env")
class TestCerebrasConfigDict:
    def test_specific_fields_included_when_set(self):
        llm = CerebrasCompletion(
            model="gpt-oss-120b",
            api_key="sk-test",
            service_tier="priority",
            prompt_cache_key="cache-1",
            clear_thinking=True,
            reasoning_effort="high",
        )
        config = llm.to_config_dict()
        assert config["service_tier"] == "priority"
        assert config["prompt_cache_key"] == "cache-1"
        assert config["clear_thinking"] is True
        assert config["reasoning_effort"] == "high"

    def test_specific_fields_omitted_when_unset(self):
        llm = CerebrasCompletion(model="gpt-oss-120b", api_key="sk-test")
        config = llm.to_config_dict()
        assert "service_tier" not in config
        assert "prompt_cache_key" not in config
        assert "clear_thinking" not in config


@pytest.mark.usefixtures("clear_cerebras_env")
class TestCerebrasCompletionParams:
    """Verify Cerebras-specific kwargs reach the chat.completions.create call."""

    def test_reasoning_effort_threaded_for_non_o1_models(self):
        llm = CerebrasCompletion(
            model="gpt-oss-120b", api_key="sk-test", reasoning_effort="high"
        )
        params = llm._prepare_completion_params(messages=[])
        # Parent gates this on is_o1_model — Cerebras must thread it regardless.
        assert params["reasoning_effort"] == "high"

    def test_service_tier_threaded(self):
        llm = CerebrasCompletion(
            model="llama3.1-8b", api_key="sk-test", service_tier="priority"
        )
        params = llm._prepare_completion_params(messages=[])
        assert params["service_tier"] == "priority"

    def test_prompt_cache_key_threaded(self):
        llm = CerebrasCompletion(
            model="llama3.1-8b", api_key="sk-test", prompt_cache_key="run-42"
        )
        params = llm._prepare_completion_params(messages=[])
        assert params["prompt_cache_key"] == "run-42"

    def test_clear_thinking_threaded(self):
        llm = CerebrasCompletion(
            model="zai-glm-4.7", api_key="sk-test", clear_thinking=True
        )
        params = llm._prepare_completion_params(messages=[])
        assert params["clear_thinking"] is True

    def test_cerebras_specific_fields_omitted_when_unset(self):
        llm = CerebrasCompletion(model="llama3.1-8b", api_key="sk-test")
        params = llm._prepare_completion_params(messages=[])
        assert "service_tier" not in params
        assert "prompt_cache_key" not in params
        assert "clear_thinking" not in params
        assert "reasoning_effort" not in params


@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_cerebras_basic_completion():
    """End-to-end completion against Cerebras (replays from cassette)."""
    llm = LLM(model="cerebras/llama3.1-8b", max_completion_tokens=32)
    assert isinstance(llm, CerebrasCompletion)

    result = llm.call("Reply with exactly the word: OK")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_cerebras_streaming_completion():
    """Streaming completion against Cerebras (replays from cassette)."""
    llm = LLM(model="cerebras/llama3.1-8b", stream=True, max_completion_tokens=32)
    assert isinstance(llm, CerebrasCompletion)

    result = llm.call("Count: one, two, three.")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr(filter_headers=["authorization", "x-api-key"])
def test_cerebras_temperature_and_seed_passed_to_sdk():
    """Deterministic-sampling params reach the Cerebras SDK call."""
    llm = LLM(
        model="cerebras/llama3.1-8b",
        temperature=0.0,
        seed=7,
        max_completion_tokens=64,
    )
    assert isinstance(llm, CerebrasCompletion)

    original_create = llm._client.chat.completions.create
    captured: dict = {}

    def capture_and_call(**kwargs):
        captured.update(kwargs)
        return original_create(**kwargs)

    with patch.object(
        llm._client.chat.completions, "create", side_effect=capture_and_call
    ):
        llm.call("Say hi.")

    assert captured["model"] == "llama3.1-8b"
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 7
