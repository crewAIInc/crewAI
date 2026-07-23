"""Tests for the model profile registry."""

import pytest

from crewai.utilities.model_profiles import (
    ModelProfile,
    _MODEL_PATTERNS,
    get_model_profile,
)


class TestModelProfile:
    """Tests for the ModelProfile dataclass."""

    def test_frozen(self):
        """ModelProfile should be immutable."""
        profile = ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.85,
            family="openai",
        )
        with pytest.raises(AttributeError):
            profile.preferred_language = "zh"  # type: ignore[misc]

    def test_hashable(self):
        """ModelProfile should be hashable (frozen dataclass)."""
        profile = ModelProfile(
            preferred_language="en",
            english_tokens_per_word=1.1,
            non_english_chars_per_token=0.6,
            bilingual_capability=0.85,
            family="openai",
        )
        # Should be usable as dict key
        d = {profile: "test"}
        assert d[profile] == "test"


class TestGetModelProfile:
    """Tests for the get_model_profile function."""

    @pytest.mark.parametrize(
        "model_name,expected_family,expected_lang",
        [
            ("gpt-4o", "openai", "en"),
            ("gpt-4o-2024-05-13", "openai", "en"),
            ("gpt-5", "openai", "en"),
            ("gpt-5-2025-08-07", "openai", "en"),
            ("claude-3-5-sonnet-20241022", "anthropic", "en"),
            ("claude-opus-4-5-20251101", "anthropic", "en"),
            ("llama-3.1-70b-versatile", "llama", "en"),
            ("llama-3.3-70b-versatile", "llama", "en"),
            ("Qwen2.5-72B-Instruct", "qwen", "zh"),
            ("deepseek-chat", "deepseek", "en"),
            ("gemini-2.5-pro", "google", "en"),
            ("gemini-2.0-flash", "google", "en"),
            ("mistral-large-latest", "mistral", "en"),
            ("mixtral-8x7b-32768", "mistral", "en"),
            ("amazon.nova-pro-v1:0", "amazon", "en"),
        ],
        ids=[
            "gpt-4o",
            "gpt-4o-versioned",
            "gpt-5",
            "gpt-5-versioned",
            "claude-sonnet",
            "claude-opus",
            "llama-3.1",
            "llama-3.3",
            "qwen",
            "deepseek",
            "gemini-pro",
            "gemini-flash",
            "mistral",
            "mixtral",
            "nova",
        ],
    )
    def test_known_models(self, model_name, expected_family, expected_lang):
        """Known models should return correct profiles."""
        profile = get_model_profile(model_name)
        assert profile is not None
        assert profile.family == expected_family
        assert profile.preferred_language == expected_lang

    def test_unknown_model_returns_none(self):
        """Unknown models should return None (conservative)."""
        assert get_model_profile("some-unknown-model-xyz") is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert get_model_profile("") is None

    def test_longest_match_wins(self):
        """Longest matching key should win for overlapping patterns."""
        profile = get_model_profile("gpt-4o-2024-05-13")
        assert profile is not None
        assert profile.family == "openai"

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        profile = get_model_profile("GPT-4O")
        assert profile is not None
        assert profile.family == "openai"

    def test_partial_model_name(self):
        """Partial model names should match if pattern matches."""
        profile = get_model_profile("my-custom-gpt-4o-deployment")
        assert profile is not None
        assert profile.family == "openai"

    def test_no_false_positive_on_unrelated_names(self):
        """Short patterns should not match unrelated model names."""
        # "o1" should NOT match "proto1" or "bio1"
        assert get_model_profile("proto1") is None
        assert get_model_profile("bio1") is None

    def test_o1_matches_with_boundary(self):
        """o1 pattern should match actual o1 models."""
        assert get_model_profile("o1-preview") is not None
        assert get_model_profile("o1-mini") is not None
        assert get_model_profile("openai/o1-preview") is not None

    def test_patterns_not_empty(self):
        """_MODEL_PATTERNS should contain entries."""
        assert len(_MODEL_PATTERNS) > 0

    def test_all_profiles_have_valid_languages(self):
        """All profiles should have valid ISO 639-1 language codes."""
        valid_langs = {
            "en", "zh", "ja", "ko", "ar", "hi", "ru", "th", "he",
        }
        for pattern, profile in _MODEL_PATTERNS:
            assert profile.preferred_language in valid_langs, (
                f"Pattern {pattern.pattern} has invalid language: "
                f"{profile.preferred_language}"
            )

    def test_all_profiles_have_valid_capability(self):
        """All profiles should have bilingual_capability between 0 and 1."""
        for pattern, profile in _MODEL_PATTERNS:
            assert 0.0 <= profile.bilingual_capability <= 1.0, (
                f"Pattern {pattern.pattern} has invalid bilingual_capability: "
                f"{profile.bilingual_capability}"
            )
