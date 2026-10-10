"""Tests for the Fireworks AI native OpenAI-compatible provider."""

import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.openai_compatible.completion import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OpenAICompatibleCompletion,
)

_FIREWORKS_MODEL = "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct"
_FIREWORKS_MODEL_ID = "accounts/fireworks/models/llama-v3p1-70b-instruct"


class TestFireworksRegistry:
    """Verify Fireworks AI's ProviderConfig in the registry."""

    def test_fireworks_ai_in_registry(self) -> None:
        assert "fireworks_ai" in OPENAI_COMPATIBLE_PROVIDERS

    def test_fireworks_ai_config(self) -> None:
        cfg = OPENAI_COMPATIBLE_PROVIDERS["fireworks_ai"]
        assert cfg.base_url == "https://api.fireworks.ai/inference/v1"
        assert cfg.api_key_env == "FIREWORKS_API_KEY"
        assert cfg.base_url_env == "FIREWORKS_BASE_URL"
        assert cfg.api_key_required is True
        assert cfg.default_api_key is None


class TestFireworksRouting:
    """Verify LLM factory routes fireworks_ai/* to OpenAICompatibleCompletion."""

    def test_prefixed_model_creates_openai_compatible(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key"}):
            llm = LLM(model=_FIREWORKS_MODEL)
        assert isinstance(llm, OpenAICompatibleCompletion)
        assert llm.provider == "fireworks_ai"

    def test_multi_slash_model_id_preserved_intact(self) -> None:
        """The Fireworks model path (accounts/fireworks/models/<id>) must reach the SDK
        with all slashes intact after the provider prefix is stripped."""
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key"}):
            llm = LLM(model=_FIREWORKS_MODEL)
        assert llm.model == _FIREWORKS_MODEL_ID

    def test_explicit_provider_parameter(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key"}):
            llm = LLM(model=_FIREWORKS_MODEL_ID, provider="fireworks_ai")
        assert isinstance(llm, OpenAICompatibleCompletion)
        assert llm.provider == "fireworks_ai"
        assert llm.model == _FIREWORKS_MODEL_ID

    def test_litellm_is_false_by_default(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key"}):
            llm = LLM(model=_FIREWORKS_MODEL)
        assert llm.is_litellm is False

    def test_is_litellm_true_forces_legacy_path(self) -> None:
        """is_litellm=True must still bypass the native provider."""
        llm = LLM(model=_FIREWORKS_MODEL, is_litellm=True)
        assert llm.is_litellm is True
        assert not isinstance(llm, OpenAICompatibleCompletion)


class TestFireworksConfiguration:
    """Verify Fireworks AI endpoint and API-key resolution precedence."""

    def test_default_base_url(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-key"}):
            llm = LLM(model=_FIREWORKS_MODEL)
        assert llm.base_url == "https://api.fireworks.ai/inference/v1"

    def test_fireworks_base_url_env_overrides_default(self) -> None:
        env = {
            "FIREWORKS_API_KEY": "test-key",
            "FIREWORKS_BASE_URL": "https://proxy.example.com/v1",
        }
        with patch.dict(os.environ, env):
            llm = LLM(model=_FIREWORKS_MODEL)
        assert llm.base_url == "https://proxy.example.com/v1"

    def test_explicit_base_url_overrides_env(self) -> None:
        env = {
            "FIREWORKS_API_KEY": "test-key",
            "FIREWORKS_BASE_URL": "https://env.example.com/v1",
        }
        with patch.dict(os.environ, env):
            llm = LLM(
                model=_FIREWORKS_MODEL,
                base_url="https://explicit.example.com/v1",
            )
        assert llm.base_url == "https://explicit.example.com/v1"

    def test_api_key_from_env(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "my-fireworks-key"}):
            llm = LLM(model=_FIREWORKS_MODEL)
        assert llm.api_key == "my-fireworks-key"

    def test_explicit_api_key_overrides_env(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "env-key"}):
            llm = LLM(model=_FIREWORKS_MODEL, api_key="explicit-key")
        assert llm.api_key == "explicit-key"

    def test_missing_api_key_raises_with_fireworks_env_var_name(self) -> None:
        """A missing key must name FIREWORKS_API_KEY, never use OPENAI_API_KEY."""
        clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("FIREWORKS_API_KEY", "OPENAI_API_KEY")
        }
        with patch.dict(os.environ, clean, clear=True), pytest.raises(
            ImportError, match="FIREWORKS_API_KEY"
        ):
            LLM(model=_FIREWORKS_MODEL)

    def test_does_not_use_openai_endpoint_on_missing_key(self) -> None:
        """Must never fall back to the OpenAI endpoint."""
        clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("FIREWORKS_API_KEY", "OPENAI_API_KEY")
        }
        with patch.dict(os.environ, clean, clear=True), pytest.raises(ImportError):
            llm = LLM(model=_FIREWORKS_MODEL)
            assert "openai.com" not in (llm.base_url or "")
