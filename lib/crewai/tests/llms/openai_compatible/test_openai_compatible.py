"""Tests for OpenAI-compatible providers."""

import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai.llms.providers.openai_compatible.completion import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OpenAICompatibleCompletion,
    ProviderConfig,
    _normalize_ollama_base_url,
)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    def test_provider_config_immutable(self):
        """Test that ProviderConfig is immutable (frozen)."""
        config = ProviderConfig(
            base_url="https://example.com/v1",
            api_key_env="TEST_API_KEY",
        )
        with pytest.raises(AttributeError):
            config.base_url = "https://other.com/v1"

    def test_provider_config_defaults(self):
        """Test ProviderConfig default values."""
        config = ProviderConfig(
            base_url="https://example.com/v1",
            api_key_env="TEST_API_KEY",
        )
        assert config.base_url_env is None
        assert config.default_headers == {}
        assert config.api_key_required is True
        assert config.default_api_key is None


class TestProviderRegistry:
    """Tests for the OPENAI_COMPATIBLE_PROVIDERS registry."""

    def test_openrouter_config(self):
        """Test OpenRouter provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["openrouter"]
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.api_key_env == "OPENROUTER_API_KEY"
        assert config.base_url_env == "OPENROUTER_BASE_URL"
        assert "HTTP-Referer" in config.default_headers
        assert config.api_key_required is True

    def test_deepseek_config(self):
        """Test DeepSeek provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["deepseek"]
        assert config.base_url == "https://api.deepseek.com/v1"
        assert config.api_key_env == "DEEPSEEK_API_KEY"
        assert config.api_key_required is True

    def test_ollama_config(self):
        """Test Ollama provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["ollama"]
        assert config.base_url == "http://localhost:11434/v1"
        assert config.api_key_env == "OLLAMA_API_KEY"
        assert config.base_url_env == "OLLAMA_HOST"
        assert config.api_key_required is False
        assert config.default_api_key == "ollama"

    def test_ollama_chat_is_alias(self):
        """Test ollama_chat is configured same as ollama."""
        ollama = OPENAI_COMPATIBLE_PROVIDERS["ollama"]
        ollama_chat = OPENAI_COMPATIBLE_PROVIDERS["ollama_chat"]
        assert ollama.base_url == ollama_chat.base_url
        assert ollama.api_key_required == ollama_chat.api_key_required

    def test_hosted_vllm_config(self):
        """Test hosted_vllm provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["hosted_vllm"]
        assert config.base_url == "http://localhost:8000/v1"
        assert config.api_key_env == "VLLM_API_KEY"
        assert config.api_key_required is False
        assert config.default_api_key == "dummy"

    def test_cerebras_config(self):
        """Test Cerebras provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["cerebras"]
        assert config.base_url == "https://api.cerebras.ai/v1"
        assert config.api_key_env == "CEREBRAS_API_KEY"
        assert config.api_key_required is True

    def test_dashscope_config(self):
        """Test Dashscope provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["dashscope"]
        assert config.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        assert config.api_key_env == "DASHSCOPE_API_KEY"
        assert config.api_key_required is True


class TestNormalizeOllamaBaseUrl:
    """Tests for _normalize_ollama_base_url helper."""

    def test_adds_v1_suffix(self):
        """Test that /v1 is added when missing."""
        assert _normalize_ollama_base_url("http://localhost:11434") == "http://localhost:11434/v1"

    def test_preserves_existing_v1(self):
        """Test that existing /v1 is preserved."""
        assert _normalize_ollama_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"

    def test_strips_trailing_slash(self):
        """Test that trailing slash is handled."""
        assert _normalize_ollama_base_url("http://localhost:11434/") == "http://localhost:11434/v1"

    def test_handles_v1_with_trailing_slash(self):
        """Test /v1/ is normalized."""
        assert _normalize_ollama_base_url("http://localhost:11434/v1/") == "http://localhost:11434/v1"


class TestOpenAICompatibleCompletion:
    """Tests for OpenAICompatibleCompletion class."""

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown OpenAI-compatible provider"):
            OpenAICompatibleCompletion(model="test", provider="unknown_provider")

    def test_missing_required_api_key_raises_error(self):
        """Test that missing required API key raises ValueError."""
        # Clear any existing env var
        env_key = "DEEPSEEK_API_KEY"
        original = os.environ.pop(env_key, None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                OpenAICompatibleCompletion(model="deepseek-chat", provider="deepseek")
        finally:
            if original is not None:
                os.environ[env_key] = original

    def test_api_key_from_env(self):
        """Test API key is read from environment variable."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key-from-env"}):
            completion = OpenAICompatibleCompletion(
                model="deepseek-chat", provider="deepseek"
            )
            assert completion.api_key == "test-key-from-env"

    def test_explicit_api_key_overrides_env(self):
        """Test explicit API key overrides environment variable."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
            completion = OpenAICompatibleCompletion(
                model="deepseek-chat",
                provider="deepseek",
                api_key="explicit-key",
            )
            assert completion.api_key == "explicit-key"

    def test_default_api_key_for_optional_providers(self):
        """Test default API key is used for providers that don't require it."""
        # Ollama doesn't require API key
        completion = OpenAICompatibleCompletion(model="llama3", provider="ollama")
        assert completion.api_key == "ollama"

    def test_base_url_from_config(self):
        """Test base URL is set from provider config."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            completion = OpenAICompatibleCompletion(
                model="deepseek-chat", provider="deepseek"
            )
            assert completion.base_url == "https://api.deepseek.com/v1"

    def test_base_url_from_env(self):
        """Test base URL is read from environment variable."""
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_BASE_URL": "https://custom.deepseek.com/v1"},
        ):
            completion = OpenAICompatibleCompletion(
                model="deepseek-chat", provider="deepseek"
            )
            assert completion.base_url == "https://custom.deepseek.com/v1"

    def test_explicit_base_url_overrides_all(self):
        """Test explicit base URL overrides env and config."""
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_BASE_URL": "https://env.deepseek.com/v1"},
        ):
            completion = OpenAICompatibleCompletion(
                model="deepseek-chat",
                provider="deepseek",
                base_url="https://explicit.deepseek.com/v1",
            )
            assert completion.base_url == "https://explicit.deepseek.com/v1"

    def test_ollama_base_url_normalized(self):
        """Test Ollama base URL is normalized to include /v1."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://custom-ollama:11434"}):
            completion = OpenAICompatibleCompletion(model="llama3", provider="ollama")
            assert completion.base_url == "http://custom-ollama:11434/v1"

    def test_openrouter_headers(self):
        """Test OpenRouter has HTTP-Referer header."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            completion = OpenAICompatibleCompletion(
                model="anthropic/claude-3-opus", provider="openrouter"
            )
            assert completion.default_headers is not None
            assert "HTTP-Referer" in completion.default_headers

    def test_custom_headers_merged_with_defaults(self):
        """Test custom headers are merged with provider defaults."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            completion = OpenAICompatibleCompletion(
                model="anthropic/claude-3-opus",
                provider="openrouter",
                default_headers={"X-Custom": "value"},
            )
            assert completion.default_headers is not None
            assert "HTTP-Referer" in completion.default_headers
            assert completion.default_headers.get("X-Custom") == "value"

    def test_supports_function_calling(self):
        """Test that function calling is supported."""
        completion = OpenAICompatibleCompletion(model="llama3", provider="ollama")
        assert completion.supports_function_calling() is True


class TestLLMIntegration:
    """Tests for LLM factory integration with OpenAI-compatible providers."""

    def test_llm_creates_openai_compatible_for_deepseek(self):
        """Test LLM factory creates OpenAICompatibleCompletion for DeepSeek."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLM(model="deepseek/deepseek-chat")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "deepseek"
            assert llm.model == "deepseek-chat"

    def test_llm_creates_openai_compatible_for_ollama(self):
        """Test LLM factory creates OpenAICompatibleCompletion for Ollama."""
        llm = LLM(model="ollama/llama3")
        assert isinstance(llm, OpenAICompatibleCompletion)
        assert llm.provider == "ollama"
        assert llm.model == "llama3"

    def test_llm_creates_openai_compatible_for_openrouter(self):
        """Test LLM factory creates OpenAICompatibleCompletion for OpenRouter."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            llm = LLM(model="openrouter/anthropic/claude-3-opus")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "openrouter"
            # Model should include the full path after provider prefix
            assert llm.model == "anthropic/claude-3-opus"

    def test_llm_creates_openai_compatible_for_hosted_vllm(self):
        """Test LLM factory creates OpenAICompatibleCompletion for hosted_vllm."""
        llm = LLM(model="hosted_vllm/meta-llama/Llama-3-8b")
        assert isinstance(llm, OpenAICompatibleCompletion)
        assert llm.provider == "hosted_vllm"

    def test_llm_creates_openai_compatible_for_cerebras(self):
        """Test LLM factory creates OpenAICompatibleCompletion for Cerebras."""
        with patch.dict(os.environ, {"CEREBRAS_API_KEY": "test-key"}):
            llm = LLM(model="cerebras/llama3-8b")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "cerebras"

    def test_llm_creates_openai_compatible_for_dashscope(self):
        """Test LLM factory creates OpenAICompatibleCompletion for Dashscope."""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            llm = LLM(model="dashscope/qwen-turbo")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "dashscope"

    def test_llm_with_explicit_provider(self):
        """Test LLM with explicit provider parameter."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLM(model="deepseek-chat", provider="deepseek")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "deepseek"
            assert llm.model == "deepseek-chat"

    def test_llm_passes_kwargs_to_completion(self):
        """Test LLM passes kwargs to OpenAICompatibleCompletion."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLM(
                model="deepseek/deepseek-chat",
                temperature=0.7,
                max_tokens=1000,
            )
            assert llm.temperature == 0.7
            assert llm.max_tokens == 1000


class TestCallMocking:
    """Tests for mocking the call method."""

    def test_call_method_can_be_mocked(self):
        """Test that the call method can be mocked for testing."""
        completion = OpenAICompatibleCompletion(model="llama3", provider="ollama")

        with patch.object(completion, "call", return_value="Mocked response"):
            result = completion.call("Test message")
            assert result == "Mocked response"

    def test_acall_method_exists(self):
        """Test that acall method exists for async calls."""
        completion = OpenAICompatibleCompletion(model="llama3", provider="ollama")
        assert hasattr(completion, "acall")
        assert callable(completion.acall)
