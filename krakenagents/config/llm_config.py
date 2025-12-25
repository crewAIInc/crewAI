"""LLM configuration for heavy/light models and embeddings."""

from functools import lru_cache
from typing import Any

from crewai import LLM

from krakenagents.config.settings import get_settings


def _get_litellm_provider(provider: str) -> str:
    """Map provider name to LiteLLM format.

    LM Studio and similar OpenAI-compatible servers use 'openai' prefix.
    """
    provider_map = {
        "lmstudio": "openai",
        "ollama": "ollama",
        "openai": "openai",
        "anthropic": "anthropic",
    }
    return provider_map.get(provider.lower(), provider)


@lru_cache
def get_heavy_llm() -> LLM:
    """Get the heavy LLM for complex reasoning tasks.

    Used for: Leadership agents, discretionary traders, research analysts.
    """
    settings = get_settings()
    provider = _get_litellm_provider(settings.llm_provider)

    return LLM(
        model=f"{provider}/{settings.llm_model}",
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key="not-needed",  # LM Studio doesn't need API key
    )


@lru_cache
def get_light_llm() -> LLM:
    """Get the light LLM for simple tasks and tool calling.

    Used for: Systematic traders, operations, execution, risk monitoring.
    """
    settings = get_settings()
    provider = _get_litellm_provider(settings.llm_light_provider)

    return LLM(
        model=f"{provider}/{settings.llm_light_model}",
        base_url=settings.llm_light_base_url,
        temperature=settings.llm_light_temperature,
        max_tokens=settings.llm_light_max_tokens,
        api_key="not-needed",  # LM Studio doesn't need API key
    )


@lru_cache
def get_chat_llm() -> LLM:
    """Get the chat LLM for conversational crew interface.

    Used for: Interactive chat with crews via dashboard.
    Uses lighter model for faster responses.
    """
    settings = get_settings()
    provider = _get_litellm_provider(settings.chat_llm_provider)

    return LLM(
        model=f"{provider}/{settings.chat_llm_model}",
        base_url=settings.chat_llm_base_url,
        temperature=settings.chat_llm_temperature,
        api_key="not-needed",  # LM Studio doesn't need API key
    )


def get_embedder_config() -> dict[str, Any]:
    """Get the embedder configuration for knowledge sources.

    Uses Ollama for local embeddings.
    """
    settings = get_settings()

    return {
        "provider": settings.embedder_provider,
        "config": {
            "model": settings.embedder_model,
            "base_url": settings.embedder_base_url,
        }
    }
