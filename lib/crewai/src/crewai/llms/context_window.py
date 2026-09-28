"""Shared context-window definitions and lookup for CrewAI LLMs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final


MIN_CONTEXT: Final[int] = 1024
MAX_CONTEXT: Final[int] = 2_097_152
DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85

# Raw provider limits. ``resolve_context_window_size`` applies the usable
# context ratio uniformly, so all call sites share one matching rule.
OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-4.1-mini-2025-04-14": 1_047_576,
    "gpt-4.1-nano-2025-04-14": 1_047_576,
    "gpt-5.4-mini": 200_000,
    "gpt-4-turbo": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5-mini": 1_047_576,
    "gpt-5-nano": 1_047_576,
    "o1-preview": 128_000,
    "gpt-5.6": 1_050_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-5": 1_047_576,
    "gpt-4": 8192,
}

AZURE_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "text-embedding": 8191,
    "gpt-3.5-turbo": 16_385,
    "gpt-35-turbo": 16_385,
}

AZURE_OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    **OPENAI_CONTEXT_WINDOWS,
    **AZURE_CONTEXT_WINDOWS,
}

ANTHROPIC_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "claude-fable-5": 1_000_000,
    "claude-mythos-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-v2:1": 200_000,
    "claude-v2": 100_000,
    "claude-instant-v1": 100_000,
}

GEMINI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gemini-3.8-flash": 1_048_576,
    "gemini-3-pro-preview": 1_048_576,
    "gemini-2.0-flash-thinking": 32_768,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-1.5-flash-8b": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.0-pro": 32_768,
    "gemma-3-27b": 128_000,
    "gemma-3-12b": 128_000,
    "gemma-3-4b": 128_000,
    "gemma-3-1b": 32_000,
}

_BEDROCK_BASE_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "amazon.nova-pro-v1:0": 300_000,
    "amazon.nova-micro-v1:0": 128_000,
    "amazon.nova-lite-v1:0": 300_000,
    "amazon.titan-text-lite-v1": 4000,
    "amazon.titan-text-express-v1": 8000,
    "cohere.command-text-v14": 4000,
    "ai21.j2-mid-v1": 8191,
    "ai21.j2-ultra-v1": 8191,
    "ai21.jamba-instruct-v1:0": 256_000,
    "mistral.mistral-7b-instruct-v0:2": 32_000,
    "mistral.mixtral-8x7b-instruct-v0:1": 32_000,
    "meta.llama3-1-405b-instruct-v1:0": 128_000,
    "meta.llama3-3-70b-instruct-v1:0": 128_000,
    "meta.llama3-1-70b-instruct-v1:0": 128_000,
    "meta.llama3-1-8b-instruct-v1:0": 128_000,
    "meta.llama3-70b-instruct-v1:0": 8000,
    "meta.llama3-8b-instruct-v1:0": 8000,
    "meta.llama3-2-11b-instruct-v1:0": 128_000,
    "meta.llama3-2-3b-instruct-v1:0": 131_000,
    "meta.llama3-2-90b-instruct-v1:0": 128_000,
    "meta.llama3-2-1b-instruct-v1:0": 131_000,
    "meta.llama2-13b-chat": 4096,
    "meta.llama2-70b-chat": 4096,
    "deepseek.r1": 32_768,
}

LITELLM_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "deepseek-chat": 128_000,
    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    "llama3-groq-70b-8192-tool-use-preview": 8192,
    "llama3-groq-8b-8192-tool-use-preview": 8192,
    "llama-3.1-70b-versatile": 131_072,
    "llama-3.1-8b-instant": 131_072,
    "llama-3.2-1b-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-11b-text-preview": 8192,
    "llama-3.2-90b-text-preview": 8192,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32_768,
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.3-70b-instruct": 128_000,
    "Meta-Llama-3.3-70B-Instruct": 131_072,
    "QwQ-32B-Preview": 8192,
    "Qwen2.5-72B-Instruct": 8192,
    "Qwen2.5-Coder-32B-Instruct": 8192,
    "Meta-Llama-3.1-405B-Instruct": 8192,
    "Meta-Llama-3.1-70B-Instruct": 131_072,
    "Meta-Llama-3.1-8B-Instruct": 131_072,
    "Llama-3.2-90B-Vision-Instruct": 16_384,
    "Llama-3.2-11B-Vision-Instruct": 16_384,
    "Meta-Llama-3.2-3B-Instruct": 4096,
    "Meta-Llama-3.2-1B-Instruct": 16_384,
    "gemini/gemma-3-1b-it": 32_000,
    "gemini/gemma-3-4b-it": 128_000,
    "gemini/gemma-3-12b-it": 128_000,
    "gemini/gemma-3-27b-it": 128_000,
    "mistral-tiny": 32_768,
    "mistral-small-latest": 32_768,
    "mistral-medium-latest": 32_768,
    "mistral-large-latest": 32_768,
    "mistral-large-2407": 32_768,
    "mistral-large-2402": 32_768,
    "mistral/mistral-tiny": 32_768,
    "mistral/mistral-small-latest": 32_768,
    "mistral/mistral-medium-latest": 32_768,
    "mistral/mistral-large-latest": 32_768,
    "mistral/mistral-large-2407": 32_768,
    "mistral/mistral-large-2402": 32_768,
}


def _prefixed_context_windows(
    sizes: Mapping[str, int], prefixes: Sequence[str]
) -> dict[str, int]:
    return {
        f"{prefix}{model}": size for prefix in prefixes for model, size in sizes.items()
    }


BEDROCK_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    **_BEDROCK_BASE_CONTEXT_WINDOWS,
    **_prefixed_context_windows(_BEDROCK_BASE_CONTEXT_WINDOWS, ("us.", "eu.", "apac.")),
    **_prefixed_context_windows(
        ANTHROPIC_CONTEXT_WINDOWS,
        (
            "anthropic.",
            "us.anthropic.",
            "eu.anthropic.",
            "apac.anthropic.",
            "global.anthropic.",
        ),
    ),
}

LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = {
    **LITELLM_CONTEXT_WINDOWS,
    **OPENAI_CONTEXT_WINDOWS,
    **AZURE_CONTEXT_WINDOWS,
    **ANTHROPIC_CONTEXT_WINDOWS,
    **GEMINI_CONTEXT_WINDOWS,
    **BEDROCK_CONTEXT_WINDOWS,
}


def resolve_context_window_size(
    model: str,
    sizes: Mapping[str, int],
    *,
    default: int,
    extra_names: Sequence[str] = (),
) -> int:
    """Return the usable context window for a model using its longest prefix."""
    for name, size in sizes.items():
        if size < MIN_CONTEXT or size > MAX_CONTEXT:
            raise ValueError(
                f"Context window for {name} must be between {MIN_CONTEXT} and {MAX_CONTEXT}"
            )

    candidates = (model, *extra_names)
    _, size = max(
        (
            (prefix, size)
            for prefix, size in sizes.items()
            if any(name.startswith(prefix) for name in candidates)
        ),
        key=lambda match: len(match[0]),
        default=("", default),
    )
    return int(size * CONTEXT_WINDOW_USAGE_RATIO)
