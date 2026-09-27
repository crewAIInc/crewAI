"""Single source of truth for LLM context window sizes.

All provider ``get_context_window_size`` implementations should call
:func:`resolve_context_window_size` with their family map instead of
hard-coding a private copy.
"""

from __future__ import annotations

from typing import Final


CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85
DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
MIN_CONTEXT: Final[int] = 1024
MAX_CONTEXT: Final[int] = 2097152


# ---------------------------------------------------------------------------
# Family maps - each provider owns only the models relevant to it.
# Keys must be sorted longest-prefix-first so that ``startswith`` matches the
# most specific entry (e.g. ``gpt-5.6`` before ``gpt-5``).
# ---------------------------------------------------------------------------

OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-4.1-mini-2025-04-14": 1_047_576,
    "gpt-4.1-nano-2025-04-14": 1_047_576,
    "gpt-5.4-mini": 200_000,
    "gpt-5.6": 1_050_000,
    "gpt-4-turbo": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5-mini": 1_047_576,
    "gpt-5-nano": 1_047_576,
    "o1-preview": 128_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-5": 1_047_576,
    "gpt-4": 8_192,
}

AZURE_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-35-turbo": 16_384,
    "text-embedding-3-small": 8_191,
    "text-embedding-3-large": 8_191,
    "text-embedding-ada-002": 8_191,
}

ANTHROPIC_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "claude-fable-5": 1_000_000,
    "claude-mythos-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-v2": 200_000,
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
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash-8b": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.0-pro": 32_768,
    "gemma-3-1b": 32_000,
    "gemma-3-4b": 128_000,
    "gemma-3-12b": 128_000,
    "gemma-3-27b": 128_000,
}

# Bedrock non-Claude models.
BEDROCK_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "amazon.nova-pro-v1:0": 300_000,
    "amazon.nova-micro-v1:0": 128_000,
    "amazon.nova-lite-v1:0": 300_000,
    "amazon.titan-text-express": 8_000,
    "amazon.titan-text-lite-v1": 4_000,
    "ai21.j2-ultra": 8_191,
    "ai21.j2-mid-v1": 8_191,
    "ai21.jamba-instruct-v1:0": 256_000,
    "cohere.command-text": 4_096,
    "meta.llama2-13b-chat": 4_096,
    "meta.llama2-70b-chat": 4_096,
    "meta.llama3-70b-instruct": 128_000,
    "meta.llama3-8b-instruct": 8_000,
    "meta.llama3-1-405b-instruct": 128_000,
    "meta.llama3-1-70b-instruct": 128_000,
    "meta.llama3-1-8b-instruct": 128_000,
    "deepseek.r1": 32_768,
    "mistral.mistral-7b-instruct": 32_000,
    "mistral.mixtral-8x7b-instruct": 32_000,
}

# LiteLLM-only models (Groq, Mistral, Together, DeepSeek, etc.) that are not
# covered by the family maps above.
LITELLM_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "deepseek-chat": 128_000,
    "deepseek/deepseek-chat": 128_000,
    "gemma2-9b-it": 8_192,
    "gemma-7b-it": 8_192,
    "llama3-groq-70b-8192-tool-use-preview": 8_192,
    "llama3-groq-8b-8192-tool-use-preview": 8_192,
    "llama-3.1-70b-versatile": 131_072,
    "llama-3.1-8b-instant": 131_072,
    "llama-3.2-1b-preview": 8_192,
    "llama-3.2-3b-preview": 8_192,
    "llama-3.2-11b-text-preview": 8_192,
    "llama-3.2-90b-text-preview": 8_192,
    "llama3-70b-8192": 8_192,
    "llama3-8b-8192": 8_192,
    "mixtral-8x7b-32768": 32_768,
    "llama-3.3-70b-versatile": 128_000,
    "llama-3.3-70b-instruct": 128_000,
    "Meta-Llama-3.3-70B-Instruct": 131_072,
    "QwQ-32B-Preview": 8_192,
    "Qwen2.5-72B-Instruct": 8_192,
    "Qwen2.5-Coder-32B-Instruct": 8_192,
    "Meta-Llama-3.1-405B-Instruct": 8_192,
    "Meta-Llama-3.1-70B-Instruct": 131_072,
    "Meta-Llama-3.1-8B-Instruct": 131_072,
    "Llama-3.2-90B-Vision-Instruct": 16_384,
    "Llama-3.2-11B-Vision-Instruct": 16_384,
    "Meta-Llama-3.2-3B-Instruct": 4_096,
    "Meta-Llama-3.2-1B-Instruct": 16_384,
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


def _expand_bedrock_claude() -> dict[str, int]:
    """Generate Bedrock Claude IDs from ``ANTHROPIC_CONTEXT_WINDOWS``.

    Bedrock prefixes Claude model names with a region scope
    (``anthropic.``, ``us.``, ``eu.``, ``apac.``, ``global.``).
    """
    prefixes = (
        "anthropic.",
        "us.anthropic.",
        "eu.anthropic.",
        "apac.anthropic.",
        "global.anthropic.",
    )
    result: dict[str, int] = {}
    for prefix in prefixes:
        for model, size in ANTHROPIC_CONTEXT_WINDOWS.items():
            result[f"{prefix}{model}"] = size
    return result


def _sort_longest_first(mapping: dict[str, int]) -> dict[str, int]:
    """Return *mapping* sorted so that longer keys come first."""
    return dict(sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True))


def resolve_context_window_size(
    model: str,
    sizes: dict[str, int],
    *,
    default: int,
    extra_names: tuple[str, ...] = (),
) -> int:
    """Resolve the usable context window for *model*.

    Longest-prefix-wins lookup over *sizes*, then apply
    :data:`CONTEXT_WINDOW_USAGE_RATIO` and clamp to
    ``[MIN_CONTEXT, MAX_CONTEXT]``.
    """
    # Validate bounds.
    for key, value in sizes.items():
        if value < MIN_CONTEXT or value > MAX_CONTEXT:
            raise ValueError(
                f"Context window for {key} must be between "
                f"{MIN_CONTEXT} and {MAX_CONTEXT}"
            )

    ordered = _sort_longest_first(sizes)
    for prefix, size in ordered.items():
        if model.startswith(prefix):
            return int(size * CONTEXT_WINDOW_USAGE_RATIO)

    # No prefix matched; try extra_names.
    for name in extra_names:
        if model.startswith(name):
            # Look up the size for this extra name.
            for prefix, size in ordered.items():
                if name.startswith(prefix):
                    return int(size * CONTEXT_WINDOW_USAGE_RATIO)
            break

    return int(default * CONTEXT_WINDOW_USAGE_RATIO)


# ---------------------------------------------------------------------------
# Merged map - used by LiteLLM fallback and exported for backwards compat.
# ---------------------------------------------------------------------------

_BEDROCK_CLAUDE_EXPANDED = _expand_bedrock_claude()

LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = _sort_longest_first(
    {
        **LITELLM_CONTEXT_WINDOWS,
        **OPENAI_CONTEXT_WINDOWS,
        **AZURE_CONTEXT_WINDOWS,
        **ANTHROPIC_CONTEXT_WINDOWS,
        **GEMINI_CONTEXT_WINDOWS,
        **BEDROCK_CONTEXT_WINDOWS,
        **_BEDROCK_CLAUDE_EXPANDED,
    }
)
