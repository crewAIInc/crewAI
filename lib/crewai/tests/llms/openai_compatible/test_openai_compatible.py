"""Tests for OpenAI-compatible providers."""

import json
import os
from unittest.mock import patch

import pytest

from crewai.llm import LLM
from crewai.llms.base_llm import MissingAPIKeyError
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

    def test_deepinfra_config(self):
        """Test DeepInfra provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["deepinfra"]
        assert config.base_url == "https://api.deepinfra.com/v1/openai"
        assert config.api_key_env == "DEEPINFRA_API_KEY"
        assert config.base_url_env == "DEEPINFRA_BASE_URL"
        assert config.api_key_required is True

    def test_nvidia_nim_config(self):
        """Test NVIDIA NIM provider configuration."""
        config = OPENAI_COMPATIBLE_PROVIDERS["nvidia_nim"]
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_env == "NVIDIA_NIM_API_KEY"
        assert config.base_url_env == "NVIDIA_NIM_BASE_URL"
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
        
    def test_bare_host_gets_scheme_and_port(self):
        """Bare host from OLLAMA_HOST gets http:// and the default port."""
        assert _normalize_ollama_base_url("0.0.0.0") == "http://0.0.0.0:11434/v1"

    def test_bare_localhost_gets_scheme_and_port(self):
        """Bare localhost gets http:// and the default port."""
        assert _normalize_ollama_base_url("localhost") == "http://localhost:11434/v1"

    def test_host_port_without_scheme_gets_scheme(self):
        """host:port without a scheme gets http:// prepended."""
        assert _normalize_ollama_base_url("127.0.0.1:11434") == "http://127.0.0.1:11434/v1"

    def test_lan_host_port_without_scheme(self):
        """A LAN host:port without a scheme gets http:// prepended."""
        assert _normalize_ollama_base_url("192.168.1.5:11434") == "http://192.168.1.5:11434/v1"

    def test_https_url_keeps_scheme_and_gets_no_default_port(self):
        """An explicit https:// URL keeps its scheme and gets no default port."""
        assert _normalize_ollama_base_url("https://ollama.example.com") == "https://ollama.example.com/v1"

    def test_root_path_with_query_does_not_double_slash(self):
        """A root path alongside a query yields /v1, not //v1."""
        assert _normalize_ollama_base_url("http://ollama/?tenant=acme") == "http://ollama:11434/v1?tenant=acme"

    def test_trailing_slash_in_query_is_preserved(self):
        """Only the path is stripped, so a query ending in / keeps that character."""
        assert _normalize_ollama_base_url("http://ollama:11434/?x=a/") == "http://ollama:11434/v1?x=a/"

class TestOpenAICompatibleCompletion:
    """Tests for OpenAICompatibleCompletion class."""

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown OpenAI-compatible provider"):
            OpenAICompatibleCompletion(model="test", provider="unknown_provider")

    def test_missing_required_api_key_raises_on_first_call(self):
        """A required key is checked on the first call, not at construction."""
        with patch.dict(os.environ, {}, clear=True):
            llm = OpenAICompatibleCompletion(model="deepseek-chat", provider="deepseek")
            with pytest.raises(
                MissingAPIKeyError,
                match="DEEPSEEK_API_KEY is required for model 'deepseek-chat'",
            ):
                llm.call("Say hi")

    def test_deepinfra_missing_api_key_names_env_var(self):
        """DeepInfra requires a key; the error tells the user which env var to set."""
        with patch.dict(os.environ, {}, clear=True):
            llm = OpenAICompatibleCompletion(
                model="deepseek-ai/DeepSeek-V4-Flash-0731", provider="deepinfra"
            )
            with pytest.raises(MissingAPIKeyError, match="DEEPINFRA_API_KEY"):
                llm.call("Say hi")

    def test_nvidia_nim_missing_api_key_names_env_var(self):
        """NVIDIA NIM requires a key; the error tells the user which env var to set."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="NVIDIA_NIM_API_KEY"
        ):
            OpenAICompatibleCompletion(
                model="nvidia/nemotron-3-super-120b-a12b", provider="nvidia_nim"
            )

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

    def test_llm_creates_openai_compatible_for_dashscope_non_qwen(self):
        """Non-Qwen DashScope models must still use the native OpenAI-compatible path."""
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "test-key",
                "DASHSCOPE_BASE_URL": "https://my-dashscope.example.com/v1",
            },
        ):
            llm = LLM(model="dashscope/deepseek-v3")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "dashscope"
            assert llm.base_url == "https://my-dashscope.example.com/v1"

    def test_llm_creates_openai_compatible_for_deepinfra(self):
        """Test LLM factory creates OpenAICompatibleCompletion for DeepInfra."""
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}):
            llm = LLM(model="deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "deepinfra"
            # DeepInfra ids are org/model, so the part after the provider
            # prefix is itself two segments and must reach the API intact.
            assert llm.model == "deepseek-ai/DeepSeek-V4-Flash-0731"
            assert llm.base_url == "https://api.deepinfra.com/v1/openai"

    def test_deepinfra_base_url_env_override(self):
        """DEEPINFRA_BASE_URL redirects DeepInfra to a proxy or private endpoint."""
        with patch.dict(
            os.environ,
            {
                "DEEPINFRA_API_KEY": "test-key",
                "DEEPINFRA_BASE_URL": "https://proxy.example.com/v1/openai",
            },
        ):
            llm = LLM(model="deepinfra/meta-llama/Llama-4-Scout-17B-16E-Instruct")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.base_url == "https://proxy.example.com/v1/openai"

    def test_llm_creates_openai_compatible_for_deepinfra_explicit_provider(self):
        """Explicit provider="deepinfra" routes natively without the model prefix."""
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}):
            llm = LLM(model="deepseek-ai/DeepSeek-V4-Flash-0731", provider="deepinfra")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "deepinfra"
            assert llm.model == "deepseek-ai/DeepSeek-V4-Flash-0731"

    def test_llm_creates_openai_compatible_for_nvidia_nim(self):
        """Test LLM factory creates OpenAICompatibleCompletion for NVIDIA NIM."""
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "test-key"}):
            llm = LLM(model="nvidia_nim/nvidia/nemotron-3-super-120b-a12b")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "nvidia_nim"
            # NIM ids are publisher/model and must reach the API intact.
            assert llm.model == "nvidia/nemotron-3-super-120b-a12b"
            assert llm.base_url == "https://integrate.api.nvidia.com/v1"

    def test_nvidia_nim_base_url_env_override(self):
        """NVIDIA_NIM_BASE_URL redirects NVIDIA NIM to a self-hosted endpoint."""
        with patch.dict(
            os.environ,
            {
                "NVIDIA_NIM_API_KEY": "test-key",
                "NVIDIA_NIM_BASE_URL": "http://localhost:8000/v1",
            },
        ):
            llm = LLM(model="nvidia_nim/nvidia/nemotron-3-super-120b-a12b")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.base_url == "http://localhost:8000/v1"

    def test_llm_creates_openai_compatible_for_nvidia_nim_explicit_provider(self):
        """Explicit provider="nvidia_nim" routes natively without the model prefix."""
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "test-key"}):
            llm = LLM(model="nvidia/nemotron-3-super-120b-a12b", provider="nvidia_nim")
            assert isinstance(llm, OpenAICompatibleCompletion)
            assert llm.provider == "nvidia_nim"
            assert llm.model == "nvidia/nemotron-3-super-120b-a12b"

    @pytest.mark.parametrize(
        "model",
        [
            "deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731",
            "DeepSeek-V4-Flash-0731",
            "deepseek-ai/DeepSeek-V4-Flash-0731/",
        ],
    )
    def test_llm_explicit_deepinfra_provider_rejects_malformed_model(self, model):
        """provider="deepinfra" applies the org/model check before building a client."""
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), pytest.raises(
            ValueError, match="org/model"
        ):
            LLM(model=model, provider="deepinfra")

    @pytest.mark.parametrize(
        "model",
        [
            "deepinfra/DeepSeek-V4-Flash-0731",
            "deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731/",
            "deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731/extra",
        ],
    )
    def test_llm_rejects_malformed_deepinfra_prefixed_model(self, model):
        """A malformed deepinfra/... id raises the org/model error, never reaching LiteLLM."""
        with patch.dict(os.environ, {"DEEPINFRA_API_KEY": "test-key"}), pytest.raises(
            ValueError, match="org/model"
        ) as exc_info:
            LLM(model=model)
        assert model in str(exc_info.value)

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("deepseek-ai/DeepSeek-V4-Flash-0731", True),
            ("Qwen/Qwen3.5-27B", True),
            ("meta-llama/Llama-4-Scout-17B-16E-Instruct", True),
            ("", False),
            ("qwen", False),
            ("Qwen/", False),
            ("/Qwen3.5-27B", False),
            ("deepseek-ai/DeepSeek-V4-Flash-0731/extra", False),
            ("deepseek-ai/DeepSeek-V4-Flash-0731/", False),
        ],
    )
    def test_deepinfra_requires_an_org_and_a_model(self, model, expected):
        """DeepInfra references are exactly org/model: one slash, both parts present."""
        assert LLM._matches_provider_pattern(model, "deepinfra") is expected

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


KEYED_PROVIDERS = {
    "openrouter": "anthropic/claude-3-opus",
    "deepseek": "deepseek-chat",
    "cerebras": "llama3.1-8b",
    "dashscope": "qwen-turbo",
    "deepinfra": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _completion(message: dict) -> dict:
    return {
        "id": "c-1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                "message": {"role": "assistant", "refusal": None, **message},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def _stream(text: str) -> bytes:
    chunks = [
        {
            "id": "c-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        }
        for part in (text[:3], text[3:])
    ]
    chunks.append(
        {
            "id": "c-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode()


class TestCallMatrixWithAKey:
    """With the key set, every call path builds its client from the provider's
    own key and URL, as before the key check moved to the first call."""

    @pytest.fixture
    def server(self, monkeypatch):
        """The real OpenAI SDK over a mock transport; records each request."""
        import httpx
        import openai

        from crewai.llms.providers.openai import completion as openai_completion

        requests: list[httpx.Request] = []
        replies: list[httpx.Response] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return replies.pop(0)

        transport = httpx.MockTransport(respond)

        def sync_client(**kwargs):
            return openai.OpenAI(**kwargs, http_client=httpx.Client(transport=transport))

        def async_client(**kwargs):
            return openai.AsyncOpenAI(
                **kwargs, http_client=httpx.AsyncClient(transport=transport)
            )

        monkeypatch.setattr(openai_completion, "OpenAI", sync_client)
        monkeypatch.setattr(openai_completion, "AsyncOpenAI", async_client)
        return requests, replies

    @staticmethod
    def build(provider: str, **kwargs) -> OpenAICompatibleCompletion:
        config = OPENAI_COMPATIBLE_PROVIDERS[provider]
        env = {config.api_key_env: f"sk-{provider}", "OPENAI_API_KEY": "sk-openai"}
        with patch.dict(os.environ, env, clear=True):
            llm = LLM(model=f"{provider}/{KEYED_PROVIDERS[provider]}", **kwargs)
        assert isinstance(llm, OpenAICompatibleCompletion)
        return llm

    @staticmethod
    def assert_sent_to(provider: str, request) -> None:
        config = OPENAI_COMPATIBLE_PROVIDERS[provider]
        assert request.headers["authorization"] == f"Bearer sk-{provider}"
        assert str(request.url).startswith(config.base_url)

    @pytest.mark.parametrize("provider", sorted(KEYED_PROVIDERS))
    @pytest.mark.parametrize("mode", ["sync", "async"])
    @pytest.mark.parametrize("stream", [False, True], ids=["no-stream", "stream"])
    def test_a_call(self, server, provider, mode, stream):
        import asyncio

        import httpx

        requests, replies = server
        replies.append(
            httpx.Response(
                200,
                content=_stream("Hello there"),
                headers={"content-type": "text/event-stream"},
            )
            if stream
            else httpx.Response(200, json=_completion({"content": "Hello there"}))
        )
        llm = self.build(provider, stream=stream)

        with patch.dict(os.environ, {}, clear=True):
            result = (
                llm.call("Say hi")
                if mode == "sync"
                else asyncio.run(llm.acall("Say hi"))
            )

        assert result == "Hello there"
        assert len(requests) == 1
        self.assert_sent_to(provider, requests[0])
        assert json.loads(requests[0].content).get("stream", False) is stream

    @pytest.mark.parametrize("provider", sorted(KEYED_PROVIDERS))
    def test_a_tool_call_then_the_follow_up_turn(self, server, provider):
        import httpx

        requests, replies = server
        replies.append(
            httpx.Response(
                200,
                json=_completion(
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Lisbon"}',
                                },
                            }
                        ],
                    }
                ),
            )
        )
        replies.append(httpx.Response(200, json=_completion({"content": "Sunny."})))
        llm = self.build(provider)
        ran: list[str] = []

        def get_weather(city: str) -> str:
            ran.append(city)
            return "sunny, 21C"

        with patch.dict(os.environ, {}, clear=True):
            tool_result = llm.call(
                "Weather in Lisbon?",
                tools=[WEATHER_TOOL],
                available_functions={"get_weather": get_weather},
            )
            follow_up = llm.call(
                [
                    {"role": "user", "content": "Weather in Lisbon?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Lisbon"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
                ],
                tools=[WEATHER_TOOL],
            )

        assert ran == ["Lisbon"]
        assert tool_result == "sunny, 21C"
        assert follow_up == "Sunny."
        assert len(requests) == 2
        for request in requests:
            self.assert_sent_to(provider, request)
        sent = json.loads(requests[1].content)["messages"]
        assert sent[-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny, 21C",
        }
