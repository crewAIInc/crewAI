"""OpenAI-compatible providers implementation.

This module provides a thin subclass of OpenAICompletion that supports
various OpenAI-compatible APIs like OpenRouter, DeepSeek, Ollama, vLLM,
Cerebras, and Dashscope (Alibaba/Qwen).

Usage:
    llm = LLM(model="deepseek/deepseek-chat")  # Uses DeepSeek API
    llm = LLM(model="openrouter/anthropic/claude-3-opus")  # Uses OpenRouter
    llm = LLM(model="ollama/llama3")  # Uses local Ollama
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

from pydantic import model_validator

from crewai.llms.providers.openai.completion import OpenAICompletion


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an OpenAI-compatible provider.

    Attributes:
        base_url: Default base URL for the provider's API endpoint.
        api_key_env: Environment variable name for the API key.
        base_url_env: Environment variable name for a custom base URL override.
        default_headers: HTTP headers to include in all requests.
        api_key_required: Whether an API key is required for this provider.
        default_api_key: Default API key to use if none is provided and not required.
    """

    base_url: str
    api_key_env: str
    base_url_env: str | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    api_key_required: bool = True
    default_api_key: str | None = None


OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        base_url_env="OPENROUTER_BASE_URL",
        default_headers={"HTTP-Referer": "https://crewai.com"},
        api_key_required=True,
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        api_key_required=True,
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        base_url_env="OLLAMA_HOST",
        api_key_required=False,
        default_api_key="ollama",
    ),
    "ollama_chat": ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        base_url_env="OLLAMA_HOST",
        api_key_required=False,
        default_api_key="ollama",
    ),
    "hosted_vllm": ProviderConfig(
        base_url="http://localhost:8000/v1",
        api_key_env="VLLM_API_KEY",
        base_url_env="VLLM_BASE_URL",
        api_key_required=False,
        default_api_key="dummy",
    ),
    "cerebras": ProviderConfig(
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        base_url_env="CEREBRAS_BASE_URL",
        api_key_required=True,
    ),
    "dashscope": ProviderConfig(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_BASE_URL",
        api_key_required=True,
    ),
}


def _normalize_ollama_base_url(base_url: str) -> str:
    """Normalize Ollama base URL to ensure it ends with /v1.

    Ollama uses OLLAMA_HOST which may not include the /v1 suffix,
    but the OpenAI-compatible endpoint requires it.

    Args:
        base_url: The base URL, potentially without /v1 suffix.

    Returns:
        The base URL with /v1 suffix if needed.
    """
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        return f"{base_url}/v1"
    return base_url


class OpenAICompatibleCompletion(OpenAICompletion):
    """OpenAI-compatible completion implementation.

    This class provides support for various OpenAI-compatible APIs by
    automatically configuring the base URL, API key, and headers based
    on the provider name.

    Supported providers:
        - openrouter: OpenRouter (https://openrouter.ai)
        - deepseek: DeepSeek (https://deepseek.com)
        - ollama: Ollama local server (https://ollama.ai)
        - ollama_chat: Alias for ollama
        - hosted_vllm: vLLM server (https://github.com/vllm-project/vllm)
        - cerebras: Cerebras (https://cerebras.ai)
        - dashscope: Alibaba Dashscope/Qwen (https://dashscope.aliyun.com)

    Example:
        # Using provider prefix
        llm = LLM(model="deepseek/deepseek-chat")

        # Using explicit provider parameter
        llm = LLM(model="llama3", provider="ollama")

        # With custom configuration
        llm = LLM(
            model="deepseek-chat",
            provider="deepseek",
            api_key="my-key",
            temperature=0.7
        )
    """

    @model_validator(mode="before")
    @classmethod
    def _resolve_provider_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        provider = data.get("provider", "")
        config = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
        if config is None:
            supported = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS.keys()))
            raise ValueError(
                f"Unknown OpenAI-compatible provider: {provider}. "
                f"Supported providers: {supported}"
            )

        data["api_key"] = cls._resolve_api_key(data.get("api_key"), config, provider)
        data["base_url"] = cls._resolve_base_url(data.get("base_url"), config, provider)
        data["default_headers"] = cls._resolve_headers(
            data.get("default_headers"), config
        )
        return data

    @staticmethod
    def _resolve_api_key(
        api_key: str | None,
        config: ProviderConfig,
        provider: str,
    ) -> str | None:
        """Resolve the API key from explicit value, env var, or default.

        Args:
            api_key: Explicitly provided API key.
            config: Provider configuration.
            provider: Provider name for error messages.

        Returns:
            The resolved API key.

        Raises:
            ValueError: If API key is required but not found.
        """
        if api_key:
            return api_key

        env_key = os.getenv(config.api_key_env)
        if env_key:
            return env_key

        if config.api_key_required:
            raise ValueError(
                f"API key required for {provider}. "
                f"Set {config.api_key_env} environment variable or pass api_key parameter."
            )

        return config.default_api_key

    @staticmethod
    def _resolve_base_url(
        base_url: str | None,
        config: ProviderConfig,
        provider: str,
    ) -> str:
        """Resolve the base URL from explicit value, env var, or default.

        Args:
            base_url: Explicitly provided base URL.
            config: Provider configuration.
            provider: Provider name (used for special handling like Ollama).

        Returns:
            The resolved base URL.
        """
        if base_url:
            resolved = base_url
        elif config.base_url_env:
            env_value = os.getenv(config.base_url_env)
            resolved = env_value if env_value else config.base_url
        else:
            resolved = config.base_url

        if provider in ("ollama", "ollama_chat"):
            resolved = _normalize_ollama_base_url(resolved)

        return resolved

    @staticmethod
    def _resolve_headers(
        headers: dict[str, str] | None,
        config: ProviderConfig,
    ) -> dict[str, str] | None:
        """Merge user headers with provider default headers.

        Args:
            headers: User-provided headers.
            config: Provider configuration.

        Returns:
            Merged headers dict, or None if empty.
        """
        if not config.default_headers and not headers:
            return None

        merged = dict(config.default_headers)
        if headers:
            merged.update(headers)

        return merged if merged else None

    def supports_function_calling(self) -> bool:
        """Check if the provider supports function calling.

        Delegates to the parent OpenAI implementation which handles
        edge cases like o1 models (which may be routed through
        OpenRouter or other compatible providers).

        Returns:
            Whether the model supports function calling.
        """
        return super().supports_function_calling()
