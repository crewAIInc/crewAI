"""Model profile registry for language-aware prompt routing.

Each model family has different tokenizer efficiency characteristics across
languages. This module maps model identifiers to profiles that describe
their language preferences and tokenizer behavior, enabling automatic
system prompt optimization.

References:
    - Multi-IF benchmark (arXiv:2410.15553): Non-Latin scripts exhibit
      systematically higher instruction-following error rates.
    - PromptQuorum (2026-05): Documents "English SP + native UP" pattern
      for DeepSeek-family models.
    - Presenc AI tokenizer benchmark (2026-05): Quantifies token cost
      asymmetry across model families.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ModelProfile:
    """Language capability profile for a model family.

    Attributes:
        preferred_language: ISO 639-1 code for the model's optimal prompt language.
        english_tokens_per_word: Average tokens per English word for this model's tokenizer.
        non_english_chars_per_token: Average non-English characters per token
            (higher = more efficient for that language).
        bilingual_capability: Score 0.0-1.0 indicating how well the model
            handles mixed-language prompts. Higher = better.
        family: Model family identifier for logging/debugging.
    """

    preferred_language: str
    english_tokens_per_word: float
    non_english_chars_per_token: float
    bilingual_capability: float
    family: str


# ---------------------------------------------------------------------------
# Registry: maps model name patterns to profiles.
# Lookup uses boundary-aware regex matching to avoid false positives.
# Unknown models return None (conservative: no translation).
# ---------------------------------------------------------------------------

# Each entry is (compiled_regex, ModelProfile).
# The regex must match at a word boundary or after a separator (/, -, _).
_MODEL_PATTERNS: list[tuple[re.Pattern[str], ModelProfile]] = [
    # --- OpenAI ---
    (
        re.compile(r"(?:^|[/_-])gpt-4o(?:[/_-]|$)"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.85,
            family="openai",
        ),
    ),
    (
        re.compile(r"(?:^|[/_-])gpt-5(?:[/_-]|$)"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.9,
            family="openai",
        ),
    ),
    (
        re.compile(r"(?:^|[/_-])o1(?:-|preview|$)"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.85,
            family="openai",
        ),
    ),
    (
        re.compile(r"(?:^|[/_-])o3(?:-|mini|$)"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.88,
            family="openai",
        ),
    ),
    (
        re.compile(r"(?:^|[/_-])o4(?:-|mini|$)"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.88,
            family="openai",
        ),
    ),
    # --- Anthropic ---
    (
        re.compile(r"(?:^|[/_-])claude"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.05,
            non_english_chars_per_token=0.85,
            bilingual_capability=0.9,
            family="anthropic",
        ),
    ),
    # --- Meta LLaMA ---
    (
        re.compile(r"(?:^|[/_-])llama"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.4,
            bilingual_capability=0.3,
            family="llama",
        ),
    ),
    # --- Qwen (Chinese-native) ---
    (
        re.compile(r"(?:^|[/_-])qwen"),
        ModelProfile(
            preferred_language="zh",
            english_tokens_per_word=1.05,
            non_english_chars_per_token=0.95,
            bilingual_capability=0.95,
            family="qwen",
        ),
    ),
    # --- DeepSeek ---
    (
        re.compile(r"(?:^|[/_-])deepseek"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.05,
            non_english_chars_per_token=0.9,
            bilingual_capability=0.85,
            family="deepseek",
        ),
    ),
    # --- Google Gemini ---
    (
        re.compile(r"(?:^|[/_-])gemini"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.8,
            bilingual_capability=0.85,
            family="google",
        ),
    ),
    # --- Mistral ---
    (
        re.compile(r"(?:^|[/_-])mistral"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.5,
            bilingual_capability=0.6,
            family="mistral",
        ),
    ),
    # --- Mixtral ---
    (
        re.compile(r"(?:^|[/_-])mixtral"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.5,
            bilingual_capability=0.6,
            family="mistral",
        ),
    ),
    # --- Amazon Nova ---
    (
        re.compile(r"(?:^|[/_-])nova"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.7,
            bilingual_capability=0.8,
            family="amazon",
        ),
    ),
    # --- Cohere ---
    (
        re.compile(r"(?:^|[/_-])command"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.5,
            bilingual_capability=0.6,
            family="cohere",
        ),
    ),
    # --- AI21 ---
    (
        re.compile(r"(?:^|[/_-])jamba"),
        ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.7,
            family="ai21",
        ),
    ),
]


def get_model_profile(model_name: str) -> ModelProfile | None:
    """Look up a model's language profile by name.

    Uses boundary-aware regex matching to avoid false positives.
    ``"gpt-4o-2024-05-13"`` matches the ``"gpt-4o"`` entry, but
    ``"proto1"`` or ``"bio1"`` will not match ``"o1"``.

    Returns ``None`` for unknown models so callers can fall back to
    conservative (no-translation) behaviour.

    Args:
        model_name: The model identifier string (e.g. ``"gpt-4o"``).

    Returns:
        The matching :class:`ModelProfile`, or ``None`` if not found.
    """
    if not model_name:
        return None

    model_lower = model_name.lower()

    # Normalize: treat `.`, `/`, `-`, `_` as equivalent separators
    normalized = re.sub(r"[./_-]", "-", model_lower)

    # Find all matching patterns and pick the longest match
    best_match: re.Match[str] | None = None
    best_profile: ModelProfile | None = None

    for pattern, profile in _MODEL_PATTERNS:
        match = pattern.search(normalized)
        if match is not None:
            # Prefer the match with the longest matched span
            if best_match is None or (match.end() - match.start()) > (
                best_match.end() - best_match.start()
            ):
                best_match = match
                best_profile = profile

    return best_profile
