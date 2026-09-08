from collections.abc import Mapping
from typing import Final


CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85

DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192

MIN_CONTEXT_WINDOW_SIZE: Final[int] = 1024

MAX_CONTEXT_WINDOW_SIZE: Final[int] = 2097152

LITELLM_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gemini-2.0-flash-thinking-exp-01-21": 32768,
    "gemini-2.0-flash-lite-001": 1048576,
    "gemini-2.0-flash-001": 1048576,
    "gemini-2.5-flash-preview-04-17": 1048576,
    "gemini-2.5-pro-exp-03-25": 1048576,
    "gemini/gemma-3-1b-it": 32000,
    "gemini/gemma-3-4b-it": 128000,
    "gemini/gemma-3-12b-it": 128000,
    "gemini/gemma-3-27b-it": 128000,
    "deepseek-chat": 128000,
    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    "llama3-groq-70b-8192-tool-use-preview": 8192,
    "llama3-groq-8b-8192-tool-use-preview": 8192,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-8b-instant": 131072,
    "llama-3.2-1b-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-11b-text-preview": 8192,
    "llama-3.2-90b-text-preview": 8192,
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    "mixtral-8x7b-32768": 32768,
    "llama-3.3-70b-versatile": 128000,
    "llama-3.3-70b-instruct": 128000,
    "Meta-Llama-3.3-70B-Instruct": 131072,
    "QwQ-32B-Preview": 8192,
    "Qwen2.5-72B-Instruct": 8192,
    "Qwen2.5-Coder-32B-Instruct": 8192,
    "Meta-Llama-3.1-405B-Instruct": 8192,
    "Meta-Llama-3.1-70B-Instruct": 131072,
    "Meta-Llama-3.1-8B-Instruct": 131072,
    "Llama-3.2-90B-Vision-Instruct": 16384,
    "Llama-3.2-11B-Vision-Instruct": 16384,
    "Meta-Llama-3.2-3B-Instruct": 4096,
    "Meta-Llama-3.2-1B-Instruct": 16384,
    "mistral-tiny": 32768,
    "mistral-small-latest": 32768,
    "mistral-medium-latest": 32768,
    "mistral-large-latest": 32768,
    "mistral-large-2407": 32768,
    "mistral-large-2402": 32768,
    "mistral/mistral-tiny": 32768,
    "mistral/mistral-small-latest": 32768,
    "mistral/mistral-medium-latest": 32768,
    "mistral/mistral-large-latest": 32768,
    "mistral/mistral-large-2407": 32768,
    "mistral/mistral-large-2402": 32768,
}

# GPT / o-series (OpenAI + Azure)
OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    "gpt-5.4-mini": 200000,
    "gpt-4-turbo": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5-mini": 1047576,
    "gpt-5-nano": 1047576,
    "o1-preview": 128000,
    "gpt-5.6": 1050000,
    "o1": 200000,
    "o1-pro": 200000,
    "o3": 200000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "o4-mini": 200000,
    "gpt-4.1": 1047576,
    "gpt-4o": 128000,
    "gpt-5": 1047576,
    "gpt-4": 8192,
}

# Azure deployments reuse the GPT map plus azure-only extras
# (gpt-3.5-turbo, text-embedding). Keep it a single source: adding a GPT
# model happens once in OPENAI_CONTEXT_WINDOWS above.
AZURE_CONTEXT_WINDOWS = {
    **OPENAI_CONTEXT_WINDOWS,
    "text-embedding": 8191,
    "gpt-3.5-turbo": 16385,
    "gpt-35-turbo": 16385,
}

# bare claude-* prefixes
ANTHROPIC_CONTEXT_WINDOWS = {
    "claude-fable-5": 1000000,
    "claude-mythos-5": 1000000,
    "claude-opus-5": 1000000,
    "claude-sonnet-5": 1000000,
    "claude-opus-4-8": 1000000,
    "claude-opus-4-7": 1000000,
    "claude-opus-4-6": 1000000,
    "claude-sonnet-4-6": 1000000,
    "claude-opus-4-5": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-haiku-4-5": 200000,
}

GEMINI_CONTEXT_WINDOWS = {
    "gemini-3-pro-preview": 1048576,  # 1M tokens
    "gemini-2.0-flash": 1048576,  # 1M tokens
    "gemini-2.0-flash-thinking": 32768,
    "gemini-2.0-flash-lite": 1048576,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-1.5-pro": 2097152,  # 2M tokens
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    "gemini-1.0-pro": 32768,
    "gemma-3-1b": 32000,
    "gemma-3-4b": 128000,
    "gemma-3-12b": 128000,
    "gemma-3-27b": 128000,
}

BEDROCK_ANTHROPIC_PREFIXES = (
    "anthropic.",
    "us.anthropic.",
    "eu.anthropic.",
    "apac.anthropic.",
    "global.anthropic.",
)

BEDROCK_ANTHROPIC_CONTEXT_WINDOWS = {
    f"{prefix}{model}": context_window
    for prefix in BEDROCK_ANTHROPIC_PREFIXES
    for model, context_window in ANTHROPIC_CONTEXT_WINDOWS.items()
}

# Titan, Nova, Llama, etc.
BEDROCK_CONTEXT_WINDOWS = {
    **BEDROCK_ANTHROPIC_CONTEXT_WINDOWS,
    "anthropic.claude-sonnet-4": 200000,
    "anthropic.claude-opus-4": 200000,
    "anthropic.claude-haiku-4": 200000,
    "anthropic.claude-3-5-sonnet": 200000,
    "anthropic.claude-3-5-haiku": 200000,
    "anthropic.claude-3-opus": 200000,
    "anthropic.claude-3-sonnet": 200000,
    "anthropic.claude-3-haiku": 200000,
    "anthropic.claude-3-7-sonnet": 200000,
    "anthropic.claude-v2": 100000,
    "amazon.titan-text-express": 8000,
    "ai21.j2-ultra": 8192,
    "cohere.command-text": 4096,
    "meta.llama2-13b-chat": 4096,
    "meta.llama2-70b-chat": 4096,
    "meta.llama3-70b-instruct": 128000,
    "deepseek.r1": 32768,
    "us.amazon.nova-pro-v1:0": 300000,
    "us.amazon.nova-micro-v1:0": 128000,
    "us.amazon.nova-lite-v1:0": 300000,
    "us.anthropic.claude-opus-4-7": 1000000,
    "us.anthropic.claude-sonnet-4-6": 1000000,
    "us.anthropic.claude-opus-4-6-v1": 1000000,
    "us.anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "us.anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "us.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "us.anthropic.claude-3-opus-20240229-v1:0": 200000,
    "us.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "us.meta.llama3-2-11b-instruct-v1:0": 128000,
    "us.meta.llama3-2-3b-instruct-v1:0": 131000,
    "us.meta.llama3-2-90b-instruct-v1:0": 128000,
    "us.meta.llama3-2-1b-instruct-v1:0": 131000,
    "us.meta.llama3-1-8b-instruct-v1:0": 128000,
    "us.meta.llama3-1-70b-instruct-v1:0": 128000,
    "us.meta.llama3-3-70b-instruct-v1:0": 128000,
    "us.meta.llama3-1-405b-instruct-v1:0": 128000,
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "eu.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "eu.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "eu.anthropic.claude-opus-4-7": 1000000,
    "eu.anthropic.claude-sonnet-4-6": 1000000,
    "eu.anthropic.claude-opus-4-6-v1": 1000000,
    "eu.anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "eu.anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "eu.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "eu.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "eu.meta.llama3-2-3b-instruct-v1:0": 131000,
    "eu.meta.llama3-2-1b-instruct-v1:0": 131000,
    "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "apac.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "apac.anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "apac.anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "apac.anthropic.claude-opus-4-7": 1000000,
    "apac.anthropic.claude-sonnet-4-6": 1000000,
    "apac.anthropic.claude-opus-4-6-v1": 1000000,
    "apac.anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "apac.anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "apac.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "apac.anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "apac.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "apac.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "amazon.nova-pro-v1:0": 300000,
    "amazon.nova-micro-v1:0": 128000,
    "amazon.nova-lite-v1:0": 300000,
    "anthropic.claude-opus-4-7": 1000000,
    "anthropic.claude-sonnet-4-6": 1000000,
    "anthropic.claude-opus-4-6-v1": 1000000,
    "anthropic.claude-opus-4-5-20251101-v1:0": 200000,
    "anthropic.claude-haiku-4-5-20251001-v1:0": 200000,
    "anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
    "anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
    "anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
    "anthropic.claude-3-opus-20240229-v1:0": 200000,
    "anthropic.claude-3-haiku-20240307-v1:0": 200000,
    "anthropic.claude-v2:1": 200000,
    "anthropic.claude-instant-v1": 100000,
    "meta.llama3-1-405b-instruct-v1:0": 128000,
    "meta.llama3-1-70b-instruct-v1:0": 128000,
    "meta.llama3-1-8b-instruct-v1:0": 128000,
    "meta.llama3-70b-instruct-v1:0": 8000,
    "meta.llama3-8b-instruct-v1:0": 8000,
    "amazon.titan-text-lite-v1": 4000,
    "amazon.titan-text-express-v1": 8000,
    "cohere.command-text-v14": 4000,
    "ai21.j2-mid-v1": 8191,
    "ai21.j2-ultra-v1": 8191,
    "ai21.jamba-instruct-v1:0": 256000,
    "mistral.mistral-7b-instruct-v0:2": 32000,
    "mistral.mixtral-8x7b-instruct-v0:1": 32000,
}


def resolve_context_window_size(
    model: str, context_windows: Mapping[str, int], *, default: int
) -> int:
    """
    Resolve the context window size for a given model.
    Model prefixes are matched using longest-prefix matching.

    Args:
      model: The model name to resolve
      context_windows: Mapping of model prefixes to raw context window sizes.
      default: Default context window size

    Returns:
      The window size which actually in use after applying the usage ratio

    Raises:
      ValueError: If the context window size is out of bounds
    """
    if default < MIN_CONTEXT_WINDOW_SIZE or default > MAX_CONTEXT_WINDOW_SIZE:
        raise ValueError(
            f"Default context window must be between {MIN_CONTEXT_WINDOW_SIZE} and {MAX_CONTEXT_WINDOW_SIZE}: default={default}"
        )
    for model_prefix, context_window in context_windows.items():
        if (
            context_window < MIN_CONTEXT_WINDOW_SIZE
            or context_window > MAX_CONTEXT_WINDOW_SIZE
        ):
            raise ValueError(
                f"Context window for {model_prefix} must be between {MIN_CONTEXT_WINDOW_SIZE} and {MAX_CONTEXT_WINDOW_SIZE}"
            )

    matches = [
        (model_prefix, context_window)
        for model_prefix, context_window in context_windows.items()
        if model.startswith(model_prefix)
    ]
    if matches:
        _, context_window = max(matches, key=lambda x: len(x[0]))
        return int(context_window * CONTEXT_WINDOW_USAGE_RATIO)
    return int(default * CONTEXT_WINDOW_USAGE_RATIO)
