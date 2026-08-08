"""Language-aware system prompt routing for multi-model agents.

Detects the language of a system prompt and translates it to the language
the target model handles best — before the prompt reaches the LLM.

This module is a pure-function pipeline with no framework coupling. It
can be tested and used independently of CrewAI's Agent class.

Design decisions:
    - **System prompt only**: User messages are never translated.
    - **Opt-in/opt-out**: ``Agent(auto_translate_prompt=False)`` disables it.
    - **Conservative thresholds**: Only translates if >10 % token savings AND
      the model has poor bilingual capability.
    - **Code block preservation**: Fenced code, inline code, URLs, and JSON
      are detected via regex and excluded from translation.
    - **Content-hash caching**: Avoids repeated translation for the same prompt.

References:
    - Multi-IF benchmark (arXiv:2410.15553)
    - PromptQuorum (2026-05)
    - Presenc AI tokenizer benchmark (2026-05)
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any
import unicodedata

from crewai.utilities.model_profiles import ModelProfile, get_model_profile


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Translation cache: content_hash -> translated text
# System prompts change infrequently, so caching avoids redundant LLM calls.
# ---------------------------------------------------------------------------
_TRANSLATION_CACHE: dict[str, str] = {}

# Minimum token savings ratio to justify translation (0.10 = 10 %)
_MIN_SAVINGS_THRESHOLD: float = 0.10

# Below this bilingual capability score, we translate; above, we leave as-is.
_BILINGUAL_THRESHOLD: float = 0.7

# ---------------------------------------------------------------------------
# Language detection via Unicode script analysis
# ---------------------------------------------------------------------------

# Unicode script ranges (approximate) for major writing systems.
_SCRIPT_RANGES: list[tuple[str, str, str]] = [
    ("zh", "\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("ja", "\u3040", "\u309f"),  # Hiragana
    ("ja", "\u30a0", "\u30ff"),  # Katakana
    ("ko", "\uac00", "\ud7af"),  # Hangul Syllables
    ("ko", "\u1100", "\u11ff"),  # Hangul Jamo
    ("ar", "\u0600", "\u06ff"),  # Arabic
    ("hi", "\u0900", "\u097f"),  # Devanagari
    ("ru", "\u0400", "\u04ff"),  # Cyrillic
    ("th", "\u0e00", "\u0e7f"),  # Thai
    ("he", "\u0590", "\u05ff"),  # Hebrew
]


def _char_script(char: str) -> str:
    """Return the script category for a single character.

    Uses Unicode name-based detection for CJK ideographs (which cover
    both Chinese and Japanese) and falls back to category analysis.
    """
    cp = ord(char)

    # Check CJK ranges first (most common non-Latin scripts)
    if 0x4E00 <= cp <= 0x9FFF:
        return "zh"  # CJK Unified Ideographs (shared by zh/ja)

    for script, start, end in _SCRIPT_RANGES:
        if ord(start) <= cp <= ord(end):
            return script

    # Latin and common punctuation / digits
    cat = unicodedata.category(char)
    if cat.startswith("L"):
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            return "en"
        if "CJK" in name:
            return "zh"

    return "other"


def detect_language(text: str) -> str:
    """Detect the primary language of *text* using Unicode script analysis.

    Returns an ISO 639-1 code: ``"en"``, ``"zh"``, ``"ja"``, ``"ko"``,
    ``"ar"``, ``"hi"``, ``"ru"``, ``"th"``, ``"he"``.

    For mixed scripts the *dominant* non-``"other"`` script wins.
    Falls back to ``"en"`` for empty input or purely numeric/punctuation text.

    Args:
        text: The text to analyse.

    Returns:
        ISO 639-1 language code.
    """
    if not text:
        return "en"

    counts: dict[str, int] = {}
    for ch in text:
        script = _char_script(ch)
        if script != "other":
            counts[script] = counts.get(script, 0) + 1

    if not counts:
        return "en"

    non_latin = {k: v for k, v in counts.items() if k != "en"}

    if not non_latin:
        return "en"

    dominant = max(non_latin, key=non_latin.get)  # type: ignore[arg-type]
    # If CJK ideographs are present and significant, decide zh vs ja
    if dominant == "zh":
        # Japanese text will also contain hiragana/katakana
        if counts.get("ja", 0) > 0:
            ja_chars = counts.get("ja", 0)
            zh_chars = non_latin.get("zh", 0)
            if ja_chars > zh_chars * 0.3:
                return "ja"
        return "zh"

    return dominant


# ---------------------------------------------------------------------------
# Token estimation (character/word heuristics — no tiktoken dependency)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z]+(?:['-][a-zA-Z]+)*")


def estimate_tokens(text: str, profile: ModelProfile) -> int:
    """Estimate the token count for *text* given a model profile.

    For English portions, words are counted and multiplied by
    ``english_tokens_per_word``.  For non-English (CJK) characters, each
    character is divided by ``non_english_chars_per_token``.

    Args:
        text: The text to estimate.
        profile: The target model's language profile.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Count English words
    english_words = len(_WORD_RE.findall(text))
    english_tokens = int(english_words * profile.english_tokens_per_word)

    # Count non-Latin characters (CJK, Cyrillic, Arabic, etc.)
    non_latin_chars = 0
    for ch in text:
        script = _char_script(ch)
        if script not in ("en", "other"):
            non_latin_chars += 1

    if profile.non_english_chars_per_token > 0:
        non_latin_tokens = int(
            non_latin_chars / profile.non_english_chars_per_token
        )
    else:
        non_latin_tokens = non_latin_chars

    return english_tokens + non_latin_tokens


# ---------------------------------------------------------------------------
# Code block / structured data extraction
# ---------------------------------------------------------------------------

# Patterns for segments that must NOT be translated
_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_URL_RE = re.compile(r"https?://\S+")
_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
_ENV_VAR_RE = re.compile(r"\b[A-Z_]{2,}=[^\s,;]+")


def extract_untranslatable_segments(
    text: str,
) -> tuple[str, list[tuple[str, str]]]:
    """Extract code blocks, URLs, JSON, and other untranslatable segments.

    Replaces each matched segment with a placeholder token so that the
    translation step only sees natural language.

    Args:
        text: The input text.

    Returns:
        A tuple of ``(text_with_placeholders, segments)`` where *segments*
        is a list of ``(placeholder, original_segment)`` pairs.
    """
    segments: list[tuple[str, str]] = []
    counter = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"__SEGMENT_{counter}__"
        segments.append((placeholder, match.group(0)))
        counter += 1
        return placeholder

    # Apply patterns in order; each pattern only matches non-overlapping text
    result = _FENCED_CODE_RE.sub(_replace, text)
    result = _INLINE_CODE_RE.sub(_replace, result)
    result = _URL_RE.sub(_replace, result)
    result = _JSON_RE.sub(_replace, result)
    result = _ENV_VAR_RE.sub(_replace, result)

    return result, segments


def restore_untranslatable_segments(
    translated: str,
    segments: list[tuple[str, str]],
) -> str:
    """Restore original untranslatable segments after translation.

    Args:
        translated: The translated text with placeholder tokens.
        segments: The ``(placeholder, original_segment)`` pairs from
            :func:`extract_untranslatable_segments`.

    Returns:
        The text with original segments restored.
    """
    result = translated
    for placeholder, original in segments:
        result = result.replace(placeholder, original)
    return result


# ---------------------------------------------------------------------------
# Translation via LLM (lightweight, cached)
# ---------------------------------------------------------------------------

# Language display names for translation prompts
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "ru": "Russian",
    "th": "Thai",
    "he": "Hebrew",
}


def _cache_key(
    text: str, lang_pair: str, glossary: dict[str, str] | None = None
) -> str:
    """Generate a content-hash cache key.

    Includes text, language pair, and glossary so different glossaries
    produce different cache entries.
    """
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(lang_pair.encode("utf-8"))
    if glossary:
        for k in sorted(glossary):
            h.update(k.encode("utf-8"))
            h.update(glossary[k].encode("utf-8"))
    return h.hexdigest()


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    glossary: dict[str, str] | None = None,
    llm_caller: Any | None = None,
) -> str:
    """Translate *text* from *source_lang* to *target_lang*.

    This function uses a cached LLM call.  If *llm_caller* is ``None``,
    it falls back to a simple glossary-apply-and-return (useful in tests
    and when no LLM is available for translation).

    Args:
        text: The text to translate.
        source_lang: ISO 639-1 source language code.
        target_lang: ISO 639-1 target language code.
        glossary: Optional dict of terms that should be preserved as-is.
        llm_caller: An object with a ``call(messages)`` method, or ``None``.

    Returns:
        The translated text, or the original text if translation fails.
    """
    if not text.strip():
        return text

    if source_lang == target_lang:
        return text

    # Check cache (key includes glossary to avoid stale translations)
    key = _cache_key(text, f"{source_lang}->{target_lang}", glossary)
    if key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[key]

    # Build glossary instruction
    glossary_instruction = ""
    if glossary:
        terms = ", ".join(f'"{k}" -> "{v}"' for k, v in glossary.items())
        glossary_instruction = (
            f"\n\nImportant: Preserve these terms exactly as written: {terms}"
        )

    source_name = _LANG_NAMES.get(source_lang, source_lang)
    target_name = _LANG_NAMES.get(target_lang, target_lang)

    prompt = (
        f"Translate the following text from {source_name} to {target_name}. "
        f"Preserve all formatting, line breaks, and placeholder tokens "
        f"(like __SEGMENT_0__). "
        f"Do not translate code, variable names, or technical identifiers."
        f"{glossary_instruction}"
        f"\n\nText to translate:\n{text}"
    )

    translated = text  # fallback

    if llm_caller is not None:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = llm_caller.call(messages)
            if isinstance(response, str) and response.strip():
                translated = response.strip()
                # Only cache successful translations — fallback text
                # should not be cached so transient failures can retry.
                _TRANSLATION_CACHE[key] = translated
        except Exception:
            logger.debug(
                "Translation LLM call failed; returning original text.",
                exc_info=True,
            )
    else:
        # No LLM available — return original (safe fallback)
        logger.debug(
            "No LLM caller provided for translation; returning original text."
        )

    return translated


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def optimize_system_prompt(
    prompt: str,
    model_name: str,
    glossary: dict[str, str] | None = None,
    llm_caller: Any | None = None,
) -> str:
    """Optimize a system prompt for the target model's language capabilities.

    Pipeline:

    1. Detect the language of the prompt.
    2. Look up the model profile.
    3. If the model prefers the same language → return as-is.
    4. Estimate token cost in the current vs. target language.
    5. If translation saves >10 % tokens AND the model has poor bilingual
       capability → translate.
    6. Extract code blocks / JSON / URLs before translation.
    7. Translate natural-language portions.
    8. Restore untranslatable segments.
    9. Cache the result.

    Args:
        prompt: The system prompt text.
        model_name: The model identifier (e.g. ``"gpt-4o"``).
        glossary: Optional dict of terms that should not be translated.
        llm_caller: An object with a ``call(messages)`` method for
            performing the actual translation, or ``None``.

    Returns:
        The (possibly translated) system prompt.
    """
    if not prompt or not prompt.strip():
        return prompt

    # 1. Detect language
    source_lang = detect_language(prompt)

    # 2. Look up model profile
    profile = get_model_profile(model_name)
    if profile is None:
        # Unknown model — conservative, no translation
        return prompt

    # 3. If model prefers this language, no translation needed
    if profile.preferred_language == source_lang:
        return prompt

    # 4. Estimate token costs
    # After translation, the text composition inverts: what was Chinese
    # becomes English words and vice versa.  We approximate the post-
    # translation cost by mapping source-language tokens to their
    # target-language equivalents.
    english_words = len(_WORD_RE.findall(prompt))
    non_latin_chars = sum(
        1 for ch in prompt if _char_script(ch) not in ("en", "other")
    )

    # Current cost (source language)
    if source_lang == "en":
        current_tokens = int(english_words * profile.english_tokens_per_word) + non_latin_chars
    else:
        current_tokens = (
            english_words
            + int(non_latin_chars / profile.non_english_chars_per_token)
        )

    # Target cost (after translation): English words in source become
    # target-lang tokens, and non-Latin chars in source become English words.
    target_lang = profile.preferred_language
    if target_lang == "en":
        # Each non-Latin char becomes ~1 English word after translation
        target_tokens = int(non_latin_chars * profile.english_tokens_per_word) + english_words
    else:
        # Each English word becomes ~1 target-lang char after translation
        target_tokens = (
            non_latin_chars
            + int(english_words / profile.non_english_chars_per_token)
        )

    # 5. Check if translation is beneficial
    if current_tokens == 0:
        return prompt

    savings = 1.0 - (target_tokens / current_tokens)

    if savings < _MIN_SAVINGS_THRESHOLD:
        return prompt

    if profile.bilingual_capability >= _BILINGUAL_THRESHOLD:
        return prompt

    # 6. Extract untranslatable segments
    cleaned_text, segments = extract_untranslatable_segments(prompt)

    # 7. Translate
    translated = translate_text(
        cleaned_text,
        source_lang=source_lang,
        target_lang=profile.preferred_language,
        glossary=glossary,
        llm_caller=llm_caller,
    )

    # 8. Restore segments
    result = restore_untranslatable_segments(translated, segments)

    # 9. Log for observability
    logger.info(
        "Prompt translated from %s to %s for model %s "
        "(estimated savings: %.1f%%)",
        source_lang,
        profile.preferred_language,
        model_name,
        savings * 100,
    )

    return result


def clear_translation_cache() -> None:
    """Clear the translation cache. Useful for testing."""
    _TRANSLATION_CACHE.clear()
