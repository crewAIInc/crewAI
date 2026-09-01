"""Regression tests for provider prefix matching (issue #5893).

Validates that self-hosted or proxy models with non-standard provider
name conventions (e.g. "anthropic--claude-...") are correctly routed.
"""

from crewai.llm import LLM


class TestMatchesProviderPattern:
    """Test LLM._matches_provider_pattern for anthropic provider."""

    def test_accepts_anthropic_double_dash_prefix(self):
        """Models like 'anthropic--claude-3.5-sonnet' should match anthropic."""
        assert LLM._matches_provider_pattern("anthropic--claude-3.5-sonnet", "anthropic") is True

    def test_accepts_anthropic_dot_prefix(self):
        """Models like 'anthropic.claude-3.5-sonnet' should still match."""
        assert LLM._matches_provider_pattern("anthropic.claude-3.5-sonnet", "anthropic") is True

    def test_accepts_claude_dash_prefix(self):
        """Models like 'claude-3.5-sonnet' should still match."""
        assert LLM._matches_provider_pattern("claude-3.5-sonnet", "anthropic") is True

    def test_rejects_unrelated_model(self):
        """Models with unrelated names should not match anthropic."""
        assert LLM._matches_provider_pattern("gpt-4o", "anthropic") is False

    def test_case_insensitive(self):
        """Prefix matching should be case-insensitive."""
        assert LLM._matches_provider_pattern("ANTHROPIC--CLAUDE-3.5", "anthropic") is True


class TestInferProviderFromModel:
    """Test LLM._infer_provider_from_model fallback logic."""

    def test_anthropic_double_dash_infers_anthropic(self):
        assert LLM._infer_provider_from_model("anthropic--claude-3.5-sonnet") == "anthropic"

    def test_claude_prefix_infers_anthropic(self):
        assert LLM._infer_provider_from_model("claude-3.5-sonnet") == "anthropic"

    def test_gemini_prefix_infers_gemini(self):
        assert LLM._infer_provider_from_model("gemini-pro") == "gemini"

    def test_gpt_prefix_infers_openai(self):
        assert LLM._infer_provider_from_model("gpt-4o") == "openai"

    def test_unknown_defaults_to_openai(self):
        assert LLM._infer_provider_from_model("some-custom-model") == "openai"

    def test_anthropic_slash_prefix_infers_anthropic(self):
        """Standard 'anthropic/' prefix also triggers anthropic inference."""
        assert LLM._infer_provider_from_model("anthropic/claude-3.5-sonnet") == "anthropic"
