"""Tests for OpenAI-compatible providers."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

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
        assert config.supports_json_schema is True

    def test_provider_config_supports_json_schema_false(self):
        """Test ProviderConfig can disable json_schema support."""
        config = ProviderConfig(
            base_url="https://example.com/v1",
            api_key_env="TEST_API_KEY",
            supports_json_schema=False,
        )
        assert config.supports_json_schema is False


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
        assert config.supports_json_schema is False

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


class TestJsonSchemaFallback:
    """Tests for json_schema fallback behavior (issue #5990).

    Providers like DeepSeek do not support json_schema response_format.
    When structured output is requested, the fallback should inject schema
    instructions into the prompt and parse the JSON response manually.
    """

    class SampleModel(BaseModel):
        name: str
        value: int

    def _make_deepseek_completion(self) -> OpenAICompatibleCompletion:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            return OpenAICompatibleCompletion(
                model="deepseek-chat", provider="deepseek"
            )

    def _make_openrouter_completion(self) -> OpenAICompatibleCompletion:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return OpenAICompatibleCompletion(
                model="anthropic/claude-3-opus", provider="openrouter"
            )

    def test_deepseek_does_not_support_json_schema(self):
        """Test that DeepSeek provider is marked as not supporting json_schema."""
        completion = self._make_deepseek_completion()
        assert completion._provider_supports_json_schema is False

    def test_openrouter_supports_json_schema(self):
        """Test that OpenRouter provider supports json_schema by default."""
        completion = self._make_openrouter_completion()
        assert completion._provider_supports_json_schema is True

    def test_prepare_params_strips_json_schema_for_deepseek(self):
        """Test that _prepare_completion_params strips json_schema response_format for DeepSeek."""
        completion = self._make_deepseek_completion()
        completion.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"a": {"type": "integer"}}},
            },
        }

        messages = [{"role": "user", "content": "test"}]
        params = completion._prepare_completion_params(messages)

        assert "response_format" not in params
        # Schema instructions should be injected into messages
        system_msgs = [m for m in params["messages"] if m["role"] == "system"]
        assert len(system_msgs) > 0
        assert "JSON schema" in system_msgs[0]["content"]

    def test_prepare_params_preserves_json_schema_for_openrouter(self):
        """Test that _prepare_completion_params preserves json_schema for OpenRouter."""
        completion = self._make_openrouter_completion()
        completion.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"a": {"type": "integer"}}},
            },
        }

        messages = [{"role": "user", "content": "test"}]
        params = completion._prepare_completion_params(messages)

        assert "response_format" in params
        assert params["response_format"]["type"] == "json_schema"

    def test_prepare_params_preserves_non_json_schema_formats(self):
        """Test that non-json_schema response_format is preserved for DeepSeek."""
        completion = self._make_deepseek_completion()
        completion.response_format = {"type": "json_object"}

        messages = [{"role": "user", "content": "test"}]
        params = completion._prepare_completion_params(messages)

        assert "response_format" in params
        assert params["response_format"]["type"] == "json_object"

    def test_handle_completion_uses_fallback_for_deepseek(self):
        """Test that _handle_completion uses fallback path for DeepSeek with response_model."""
        completion = self._make_deepseek_completion()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"name": "test", "value": 42})
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(completion, "_get_sync_client", return_value=mock_client):
            result = completion._handle_completion(
                params={"messages": [{"role": "user", "content": "test"}], "model": "deepseek-chat"},
                response_model=self.SampleModel,
            )

        # Should have used regular create, not beta.chat.completions.parse
        mock_client.chat.completions.create.assert_called_once()
        mock_client.beta.chat.completions.parse.assert_not_called()

        assert isinstance(result, self.SampleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_handle_completion_delegates_to_parent_for_openrouter(self):
        """Test that _handle_completion delegates to parent for OpenRouter with response_model."""
        completion = self._make_openrouter_completion()

        mock_parsed = MagicMock()
        mock_parsed.choices = [MagicMock()]
        mock_parsed.choices[0].message.parsed = self.SampleModel(name="test", value=42)
        mock_parsed.choices[0].message.refusal = None
        mock_parsed.usage = MagicMock()
        mock_parsed.usage.prompt_tokens = 10
        mock_parsed.usage.completion_tokens = 5
        mock_parsed.usage.total_tokens = 15

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = mock_parsed

        with patch.object(completion, "_get_sync_client", return_value=mock_client):
            result = completion._handle_completion(
                params={"messages": [{"role": "user", "content": "test"}], "model": "claude-3-opus"},
                response_model=self.SampleModel,
            )

        # Should have used beta.chat.completions.parse
        mock_client.beta.chat.completions.parse.assert_called_once()
        assert isinstance(result, self.SampleModel)

    def test_handle_completion_no_response_model_delegates_to_parent(self):
        """Test that _handle_completion delegates to parent when no response_model."""
        completion = self._make_deepseek_completion()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(completion, "_get_sync_client", return_value=mock_client):
            result = completion._handle_completion(
                params={"messages": [{"role": "user", "content": "hi"}], "model": "deepseek-chat"},
                response_model=None,
            )

        assert result == "Hello!"

    def test_handle_completion_fallback_with_markdown_wrapped_json(self):
        """Test fallback parsing handles JSON wrapped in markdown code blocks."""
        completion = self._make_deepseek_completion()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"name": "test", "value": 99}\n```'
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch.object(completion, "_get_sync_client", return_value=mock_client):
            result = completion._handle_completion(
                params={"messages": [{"role": "user", "content": "test"}], "model": "deepseek-chat"},
                response_model=self.SampleModel,
            )

        assert isinstance(result, self.SampleModel)
        assert result.name == "test"
        assert result.value == 99

    def test_inject_schema_instructions_appends_to_existing_system_message(self):
        """Test that schema instructions are appended to existing system message."""
        completion = self._make_deepseek_completion()

        params = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "test"},
            ]
        }
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        completion._inject_schema_instructions(params, schema)

        system_msg = params["messages"][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"].startswith("You are helpful.")
        assert "JSON schema" in system_msg["content"]

    def test_inject_schema_instructions_adds_system_message_if_missing(self):
        """Test that a system message is created when none exists."""
        completion = self._make_deepseek_completion()

        params = {
            "messages": [
                {"role": "user", "content": "test"},
            ]
        }
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        completion._inject_schema_instructions(params, schema)

        assert params["messages"][0]["role"] == "system"
        assert "JSON schema" in params["messages"][0]["content"]
        assert params["messages"][1]["role"] == "user"

    def test_extract_json_from_plain_text(self):
        """Test extracting JSON from plain text."""
        text = '{"name": "test", "value": 1}'
        assert OpenAICompatibleCompletion._extract_json_from_text(text) == text

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from a markdown code block."""
        text = '```json\n{"name": "test", "value": 1}\n```'
        result = OpenAICompatibleCompletion._extract_json_from_text(text)
        assert result == '{"name": "test", "value": 1}'

    def test_streaming_completion_uses_fallback_for_deepseek(self):
        """Test streaming completion uses fallback for DeepSeek with response_model."""
        completion = self._make_deepseek_completion()

        chunk1 = MagicMock()
        chunk1.id = "test-id"
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = '{"name": "stream"'
        chunk1.choices[0].delta.tool_calls = None
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.id = "test-id"
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = ', "value": 7}'
        chunk2.choices[0].delta.tool_calls = None
        chunk2.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

        with patch.object(completion, "_get_sync_client", return_value=mock_client):
            result = completion._handle_streaming_completion(
                params={
                    "messages": [{"role": "user", "content": "test"}],
                    "model": "deepseek-chat",
                    "stream": True,
                },
                response_model=self.SampleModel,
            )

        # Should have used regular create with stream, not beta.chat.completions.stream
        mock_client.chat.completions.create.assert_called_once()
        assert isinstance(result, self.SampleModel)
        assert result.name == "stream"
        assert result.value == 7
