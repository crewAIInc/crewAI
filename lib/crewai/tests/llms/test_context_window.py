"""Tests for crewai.llms.context_window."""

from __future__ import annotations

import pytest

from crewai.llms.context_window import (
    ANTHROPIC_CONTEXT_WINDOWS,
    BEDROCK_CONTEXT_WINDOWS,
    CONTEXT_WINDOW_USAGE_RATIO,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    GEMINI_CONTEXT_WINDOWS,
    LITELLM_CONTEXT_WINDOWS,
    OPENAI_CONTEXT_WINDOWS,
    LLM_CONTEXT_WINDOW_SIZES,
    _expand_bedrock_claude,
    resolve_context_window_size,
)


class TestResolveContextWindowSize:
    def test_longest_prefix_wins(self) -> None:
        sizes = {"gpt-5": 100_000, "gpt-5.6": 200_000}
        assert (
            resolve_context_window_size("gpt-5.6-luna", sizes, default=8_192)
            == int(200_000 * CONTEXT_WINDOW_USAGE_RATIO)
        )

    def test_shorter_prefix_also_matches(self) -> None:
        sizes = {"gpt-5": 100_000, "gpt-5.6": 200_000}
        assert (
            resolve_context_window_size("gpt-5-turbo", sizes, default=8_192)
            == int(100_000 * CONTEXT_WINDOW_USAGE_RATIO)
        )

    def test_no_match_returns_default(self) -> None:
        sizes = {"gpt-5": 100_000}
        assert (
            resolve_context_window_size("claude-3", sizes, default=8_192)
            == int(8_192 * CONTEXT_WINDOW_USAGE_RATIO)
        )

    def test_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="must be between 1024 and 2097152"):
            resolve_context_window_size("m", {"m": 500}, default=8_192)

    def test_exact_match(self) -> None:
        sizes = {"gpt-4o": 128_000}
        assert (
            resolve_context_window_size("gpt-4o", sizes, default=8_192)
            == int(128_000 * CONTEXT_WINDOW_USAGE_RATIO)
        )


class TestParity:
    """Native vs LiteLLM should return the same window for the same model."""

    @pytest.mark.parametrize(
        "model, family_map, expected_raw",
        [
            ("gpt-5", OPENAI_CONTEXT_WINDOWS, 1_047_576),
            ("gpt-5.6", OPENAI_CONTEXT_WINDOWS, 1_050_000),
            ("claude-sonnet-4-6", ANTHROPIC_CONTEXT_WINDOWS, 1_000_000),
            ("gemini-2.0-flash", GEMINI_CONTEXT_WINDOWS, 1_048_576),
        ],
    )
    def test_native_matches_litellm(
        self, model: str, family_map: dict[str, int], expected_raw: int
    ) -> None:
        native = resolve_context_window_size(model, family_map, default=8_192)
        litellm = resolve_context_window_size(
            model, LLM_CONTEXT_WINDOW_SIZES, default=DEFAULT_CONTEXT_WINDOW_SIZE
        )
        assert native == litellm == int(expected_raw * CONTEXT_WINDOW_USAGE_RATIO)

    def test_litellm_gpt5_uses_full_window(self) -> None:
        """gpt-5 via LiteLLM should NOT fall through to 8192."""
        result = resolve_context_window_size(
            "gpt-5", LLM_CONTEXT_WINDOW_SIZES, default=DEFAULT_CONTEXT_WINDOW_SIZE
        )
        assert result == int(1_047_576 * CONTEXT_WINDOW_USAGE_RATIO)


class TestBedrockExpansion:
    def test_bedrock_claude_expansion(self) -> None:
        expanded = _expand_bedrock_claude()
        # All regional prefixes should be present
        assert "anthropic.claude-sonnet-4-6" in expanded
        assert "us.anthropic.claude-sonnet-4-6" in expanded
        assert "eu.anthropic.claude-sonnet-4-6" in expanded
        assert "apac.anthropic.claude-sonnet-4-6" in expanded
        assert "global.anthropic.claude-sonnet-4-6" in expanded
        # Value should match the bare Anthropic entry
        assert (
            expanded["anthropic.claude-sonnet-4-6"]
            == ANTHROPIC_CONTEXT_WINDOWS["claude-sonnet-4-6"]
        )

    def test_bedrock_sonnet_4_6_is_1m_not_200k(self) -> None:
        expanded = _expand_bedrock_claude()
        assert expanded["anthropic.claude-sonnet-4-6"] == 1_000_000


class TestBackwardsCompatExports:
    def test_llm_context_window_sizes_is_merged(self) -> None:
        """LLM_CONTEXT_WINDOW_SIZES should contain entries from all family maps."""
        assert len(LLM_CONTEXT_WINDOW_SIZES) > len(OPENAI_CONTEXT_WINDOWS)
        assert len(LLM_CONTEXT_WINDOW_SIZES) > len(ANTHROPIC_CONTEXT_WINDOWS)

    def test_llm_reexports(self) -> None:
        from crewai.llm import (
            CONTEXT_WINDOW_USAGE_RATIO,
            DEFAULT_CONTEXT_WINDOW_SIZE,
            LLM_CONTEXT_WINDOW_SIZES,
        )

        assert CONTEXT_WINDOW_USAGE_RATIO == 0.85
        assert DEFAULT_CONTEXT_WINDOW_SIZE == 8_192
        assert len(LLM_CONTEXT_WINDOW_SIZES) > 0
