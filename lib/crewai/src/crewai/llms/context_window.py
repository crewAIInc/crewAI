"""Shared context-window definitions and lookup for CrewAI LLMs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final


MIN_CONTEXT: Final[int] = 1024
MAX_CONTEXT: Final[int] = 10_000_000
DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85

# Raw provider limits. ``resolve_context_window_size`` applies the usable
# context ratio uniformly, so all call sites share one matching rule.
# Keep active IDs and context windows in sync with OpenAI's model catalog;
# remove or replace entries when the vendor marks them retired:
# https://developers.openai.com/api/docs/models
OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-4.1-mini-2025-04-14": 1_047_576,
    "gpt-4.1-nano-2025-04-14": 1_047_576,
    "gpt-6": 1_050_000,
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.4-nano": 400_000,
    "gpt-5.3-codex": 400_000,
    "gpt-4-turbo": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5.6": 1_050_000,
    "o1-preview": 128_000,
    "o1-mini": 128_000,
    "o1-pro": 200_000,
    "o1": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-5": 400_000,
    "gpt-4": 8192,
}

# Check Azure deployment IDs and context windows here; availability varies by
# account and region. Remove or replace vendor-retired entries:
# https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure
AZURE_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-6-astra": 1_050_000,
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    "gpt-chat-latest": 400_000,
    "gpt-5.4-pro": 1_050_000,
    "gpt-5.2-codex": 400_000,
    "gpt-5.2": 400_000,
    "gpt-5.1-codex-mini": 400_000,
    "gpt-5.1-codex-max": 400_000,
    "gpt-5.1-codex": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5-codex": 400_000,
    "gpt-5-pro": 400_000,
    "gpt-oss-120b": 131_072,
    "gpt-oss-20b": 131_072,
    "codex-mini": 200_000,
    "o3-pro": 200_000,
    "computer-use-preview": 8192,
    "gpt-4": 128_000,
    "text-embedding-3-large": 8192,
    "text-embedding-3-small": 8192,
    "text-embedding-ada-002": 8192,
    "text-embedding": 8191,
}

AZURE_OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    **OPENAI_CONTEXT_WINDOWS,
    **AZURE_CONTEXT_WINDOWS,
}

# Check active Claude IDs and context windows here; review retirements before
# removing legacy prefixes: https://docs.anthropic.com/en/docs/about-claude/models
# https://docs.anthropic.com/en/docs/about-claude/model-deprecations
ANTHROPIC_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "claude-fable-5-1": 1_000_000,
    "claude-fable-5": 1_000_000,
    "claude-mythos-5-1": 1_000_000,
    "claude-mythos-5": 1_000_000,
    "claude-opus-5-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
}

# Check Gemini model IDs and context windows here; remove or replace retired
# entries listed in the catalog:
# https://ai.google.dev/gemini-api/docs/models
GEMINI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gemini-3.8-flash": 1_048_576,
    "gemini-3.7-flash": 1_048_576,
    "gemini-3.6-flash": 1_048_576,
    "gemini-3.5-flash": 1_048_576,
    "gemini-3.5-flash-lite": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.0-flash-thinking-exp-01-21": 1_048_576,
    "gemini-2.0-flash-thinking": 32_768,
    "gemini-1.0-pro": 32_768,
    "gemma-3-27b": 128_000,
    "gemma-3-12b": 128_000,
    "gemma-3-4b": 128_000,
    "gemma-3-1b": 32_000,
}

# Check Bedrock model IDs and each model's context window in Models at a glance.
# Availability varies by account and region; remove or replace retired entries:
# https://docs.aws.amazon.com/bedrock/latest/userguide/models.html
_BEDROCK_BASE_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "amazon.nova-2-lite-v1:0": 1_000_000,
    "amazon.nova-premier-v1:0": 1_000_000,
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
    "meta.llama4-scout-17b-instruct-v1:0": 10_000_000,
    "meta.llama2-13b-chat": 4096,
    "meta.llama2-70b-chat": 4096,
    "deepseek.r1": 32_768,
    "openai.gpt-oss-20b-1:0": 128_000,
    "openai.gpt-oss-120b-1:0": 128_000,
}

# Bedrock's lifecycle is independent from Anthropic's direct API. Sonnet 4 is
# Legacy (not EOL) on Bedrock, so keep its provider-specific limit until AWS
# retires it: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html
_BEDROCK_LEGACY_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "anthropic.claude-sonnet-4": 200_000,
    "us.anthropic.claude-sonnet-4": 200_000,
    "eu.anthropic.claude-sonnet-4": 200_000,
    "apac.anthropic.claude-sonnet-4": 200_000,
    "global.anthropic.claude-sonnet-4": 200_000,
}

# LiteLLM maintains its provider-only model IDs and context windows here. Prefer
# the vendor catalogs above when both sources define the same model:
# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
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
    **_BEDROCK_LEGACY_CONTEXT_WINDOWS,
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
