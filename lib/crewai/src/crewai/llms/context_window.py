from typing import Final, Mapping, Sequence

MIN_CONTEXT: Final[int] = 1024
MAX_CONTEXT: Final[int] = 2097152  # Current max from gemini-1.5-pro

DEFAULT_CONTEXT_WINDOW_SIZE: Final[int] = 8192
CONTEXT_WINDOW_USAGE_RATIO: Final[float] = 0.85

OPENAI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5": 128000,
    "gpt-5.4-mini": 200000,
    "gpt-5.6": 1050000,
    "gpt-4-turbo": 128000,
    "gpt-4.1": 1047576,
    "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 200000,
    "o4-mini": 200000,
}

AZURE_CONTEXT_WINDOWS: Final[dict[str, int]] = {}

ANTHROPIC_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "claude-opus-4-7": 1000000,
    "claude-sonnet-4-6": 1000000,
    "claude-opus-4-6-v1": 1000000,
    "claude-opus-4-5-20251101-v1:0": 200000,
    "claude-haiku-4-5-20251001-v1:0": 200000,
    "claude-sonnet-4-5-20250929-v1:0": 200000,
    "claude-opus-4-1-20250805-v1:0": 200000,
    "claude-opus-4-20250514-v1:0": 200000,
    "claude-sonnet-4-20250514-v1:0": 200000,
    "claude-3-5-sonnet-20240620-v1:0": 200000,
    "claude-3-5-haiku-20241022-v1:0": 200000,
    "claude-3-5-sonnet-20241022-v2:0": 200000,
    "claude-3-7-sonnet-20250219-v1:0": 200000,
    "claude-3-sonnet-20240229-v1:0": 200000,
    "claude-3-opus-20240229-v1:0": 200000,
    "claude-3-haiku-20240307-v1:0": 200000,
    "claude-v2:1": 200000,
    "claude-v2": 100000,
    "claude-instant-v1": 100000,
}

GEMINI_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "gemini-3-pro-preview": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-thinking-exp-01-21": 32768,
    "gemini-2.0-flash-lite-001": 1048576,
    "gemini-2.0-flash-001": 1048576,
    "gemini-2.5-flash-preview-04-17": 1048576,
    "gemini-2.5-pro-exp-03-25": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    "gemma-3-1b-it": 32000,
    "gemma-3-4b-it": 128000,
    "gemma-3-12b-it": 128000,
    "gemma-3-27b-it": 128000,
}

BEDROCK_CONTEXT_WINDOWS: Final[dict[str, int]] = {
    "amazon.nova-pro-v1:0": 300000,
    "amazon.nova-micro-v1:0": 128000,
    "amazon.nova-lite-v1:0": 300000,
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

# Add generated Bedrock prefixes for Claude
_bedrock_claude = {}
for _claude_key, _claude_val in ANTHROPIC_CONTEXT_WINDOWS.items():
    for _prefix in ("anthropic.", "us.anthropic.", "eu.anthropic.", "apac.anthropic.", "global.anthropic."):
        _bedrock_claude[f"{_prefix}{_claude_key}"] = _claude_val
BEDROCK_CONTEXT_WINDOWS.update(_bedrock_claude)

LITELLM_CONTEXT_WINDOWS: Final[dict[str, int]] = {
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
    "us.meta.llama3-2-11b-instruct-v1:0": 128000,
    "us.meta.llama3-2-3b-instruct-v1:0": 131000,
    "us.meta.llama3-2-90b-instruct-v1:0": 128000,
    "us.meta.llama3-2-1b-instruct-v1:0": 131000,
    "us.meta.llama3-1-8b-instruct-v1:0": 128000,
    "us.meta.llama3-1-70b-instruct-v1:0": 128000,
    "us.meta.llama3-3-70b-instruct-v1:0": 128000,
    "eu.meta.llama3-2-3b-instruct-v1:0": 131000,
    "eu.meta.llama3-2-1b-instruct-v1:0": 131000,
}

# Add gemini prefix to GEMINI_CONTEXT_WINDOWS items
_gemini_prefixed = {}
for _k, _v in GEMINI_CONTEXT_WINDOWS.items():
    _gemini_prefixed[f"gemini/{_k}"] = _v
    _gemini_prefixed[_k] = _v
GEMINI_CONTEXT_WINDOWS = _gemini_prefixed

LLM_CONTEXT_WINDOW_SIZES: Final[dict[str, int]] = {
    **OPENAI_CONTEXT_WINDOWS,
    **AZURE_CONTEXT_WINDOWS,
    **ANTHROPIC_CONTEXT_WINDOWS,
    **GEMINI_CONTEXT_WINDOWS,
    **BEDROCK_CONTEXT_WINDOWS,
    **LITELLM_CONTEXT_WINDOWS,
}

def resolve_context_window_size(
    model: str,
    sizes: Mapping[str, int],
    *,
    default: int = DEFAULT_CONTEXT_WINDOW_SIZE,
    extra_names: Sequence[str] = ()
) -> int:
    """Resolve the context window size for a model based on longest-prefix matching.

    Args:
        model: The model name to look up.
        sizes: A mapping of model prefixes to their context window sizes.
        default: The default size to return if no prefix matches.
        extra_names: Additional model names to check (e.g., bare names without provider prefix).
    """
    candidates = [model] + list(extra_names)
    
    # Find all matches across all candidates
    matches = []
    for candidate in candidates:
        for prefix, size in sizes.items():
            if candidate.startswith(prefix):
                matches.append((len(prefix), size))
                
    if not matches:
        return int(default * CONTEXT_WINDOW_USAGE_RATIO)
        
    # Longest prefix wins
    matches.sort(key=lambda x: x[0], reverse=True)
    best_size = matches[0][1]
    
    # Bound the size
    bounded_size = min(MAX_CONTEXT, max(MIN_CONTEXT, best_size))
    return int(bounded_size * CONTEXT_WINDOW_USAGE_RATIO)
