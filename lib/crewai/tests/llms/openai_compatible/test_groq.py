"""Tests for the Groq native OpenAI-compatible provider."""

import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.openai_compatible.completion import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OpenAICompatibleCompletion,
)


class TestGroqRegistry:
    """Verify Groq's ProviderConfig in the registry."""

    def test_groq_in_registry(self) -> None:
        assert "groq" in OPENAI_COMPATIBLE_PROVIDERS

    def test_groq_config(self) -> None:
        cfg = OPENAI_COMPATIBLE_PROVIDERS["groq"]
        assert cfg.base_url == "https://api.groq.com/openai/v1"
        assert cfg.api_key_env == "GROQ_API_KEY"
        assert cfg.base_url_env == "GROQ_BASE_URL"
        assert cfg.api_key_required is True
        assert cfg.default_api_key is None


class TestGroqRouting:
    """Verify LLM factory routes groq/* to OpenAICompatibleCompletion."""

    def test_prefixed_model_creates_openai_compatible(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            llm = LLM(model="groq/llama-3.1-70b-versatile")
        assert isinstance(llm, OpenAICompatibleCompletion)
        assert llm.provider == "groq"

    def test_model_id_stripped_of_prefix(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            llm = LLM(model="groq/llama-3.3-70b-versatile")
        assert llm.model == "llama-3.3-70b-versatile"

    def test_explicit_provider_parameter(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            llm = LLM(model="llama-3.1-8b-instant", provider="groq")
        assert isinstance(llm, OpenAICompatibleCompletion)
        assert llm.provider == "groq"
        assert llm.model == "llama-3.1-8b-instant"

    def test_litellm_is_false_by_default(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            llm = LLM(model="groq/llama-3.1-70b-versatile")
        assert llm.is_litellm is False

    def test_is_litellm_true_forces_legacy_path(self) -> None:
        """is_litellm=True must still bypass the native provider."""
        llm = LLM(model="groq/llama-3.1-70b-versatile", is_litellm=True)
        assert llm.is_litellm is True
        assert not isinstance(llm, OpenAICompatibleCompletion)


class TestGroqConfiguration:
    """Verify Groq endpoint and API-key resolution precedence."""

    def test_default_base_url(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            llm = LLM(model="groq/llama-3.1-70b-versatile")
        assert llm.base_url == "https://api.groq.com/openai/v1"

    def test_groq_base_url_env_overrides_default(self) -> None:
        env = {"GROQ_API_KEY": "test-key", "GROQ_BASE_URL": "https://proxy.example.com/v1"}
        with patch.dict(os.environ, env):
            llm = LLM(model="groq/llama-3.1-70b-versatile")
        assert llm.base_url == "https://proxy.example.com/v1"

    def test_explicit_base_url_overrides_env(self) -> None:
        env = {"GROQ_API_KEY": "test-key", "GROQ_BASE_URL": "https://env.example.com/v1"}
        with patch.dict(os.environ, env):
            llm = LLM(
                model="groq/llama-3.1-70b-versatile",
                base_url="https://explicit.example.com/v1",
            )
        assert llm.base_url == "https://explicit.example.com/v1"

    def test_api_key_from_env(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "my-groq-key"}):
            llm = LLM(model="groq/llama-3.1-70b-versatile")
        assert llm.api_key == "my-groq-key"

    def test_explicit_api_key_overrides_env(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "env-key"}):
            llm = LLM(model="groq/llama-3.1-70b-versatile", api_key="explicit-key")
        assert llm.api_key == "explicit-key"

    def test_missing_api_key_raises_with_groq_env_var_name(self) -> None:
        """A missing key must name GROQ_API_KEY, never silently use OPENAI_API_KEY."""
        clean = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, clean, clear=True), pytest.raises(ImportError, match="GROQ_API_KEY"):
            LLM(model="groq/llama-3.1-70b-versatile")

    def test_does_not_use_openai_endpoint_on_missing_key(self) -> None:
        """Must never fall back to the OpenAI endpoint."""
        clean = {k: v for k, v in os.environ.items() if k not in ("GROQ_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, clean, clear=True), pytest.raises(ImportError):
            llm = LLM(model="groq/llama-3.1-70b-versatile")
            # If construction somehow succeeds, the endpoint must not be OpenAI's
            assert "openai.com" not in (llm.base_url or "")
