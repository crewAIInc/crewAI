"""Tests for native Groq provider support and cache_breakpoint stripping."""

import pytest


class TestGroqNativeRouting:
    """Test that Groq models route to the native OpenAI-compatible provider."""

    def test_groq_in_supported_providers(self):
        from crewai.llm import SUPPORTED_NATIVE_PROVIDERS

        assert "groq" in SUPPORTED_NATIVE_PROVIDERS

    def test_groq_provider_config_exists(self):
        from crewai.llms.providers.openai_compatible.completion import (
            OPENAI_COMPATIBLE_PROVIDERS,
        )

        assert "groq" in OPENAI_COMPATIBLE_PROVIDERS
        config = OPENAI_COMPATIBLE_PROVIDERS["groq"]
        assert config.base_url == "https://api.groq.com/openai/v1"
        assert config.api_key_env == "GROQ_API_KEY"
        assert config.api_key_required is True

    def test_groq_model_pattern_matching(self):
        from crewai.llm import LLM

        assert LLM._matches_provider_pattern("llama-3.3-70b-versatile", "groq") is True
        assert LLM._matches_provider_pattern("mixtral-8x7b-32768", "groq") is True
        assert LLM._matches_provider_pattern("gemma-7b-it", "groq") is True
        assert LLM._matches_provider_pattern("whisper-large-v3", "groq") is True
        assert (
            LLM._matches_provider_pattern("deepseek-r1-distill-llama-70b", "groq")
            is True
        )
        assert LLM._matches_provider_pattern("gpt-4o", "groq") is False

    def test_groq_routes_to_openai_compatible(self):
        from crewai.llm import LLM
        from crewai.llms.providers.openai_compatible.completion import (
            OpenAICompatibleCompletion,
        )

        provider_class = LLM._get_native_provider("groq")
        assert provider_class is OpenAICompatibleCompletion


class TestCacheBreakpointStripping:
    """Test that cache_breakpoint is stripped for non-Anthropic providers."""

    def test_strip_cache_breakpoint_for_non_anthropic(self):
        from crewai.llm import LLM

        llm = LLM.__new__(LLM, model="groq/llama-3.3-70b-versatile")
        llm.model = "groq/llama-3.3-70b-versatile"
        llm.is_anthropic = False

        messages = [
            {"role": "system", "content": "You are helpful.", "cache_breakpoint": True},
            {"role": "user", "content": "Hello", "cache_breakpoint": True},
        ]

        result = llm._format_messages_for_provider(messages)

        for msg in result:
            assert "cache_breakpoint" not in msg

    def test_preserve_cache_breakpoint_for_anthropic(self):
        from crewai.llm import LLM

        llm = LLM.__new__(LLM, model="anthropic/claude-sonnet-4-20250514")
        llm.model = "anthropic/claude-sonnet-4-20250514"
        llm.is_anthropic = True

        messages = [
            {"role": "user", "content": "Hello", "cache_breakpoint": True},
        ]

        result = llm._format_messages_for_provider(messages)

        assert result[0].get("cache_breakpoint") is True

    def test_strip_does_not_remove_role_or_content(self):
        from crewai.llm import LLM

        llm = LLM.__new__(LLM, model="groq/llama-3.3-70b-versatile")
        llm.model = "groq/llama-3.3-70b-versatile"
        llm.is_anthropic = False

        messages = [
            {"role": "user", "content": "Test message", "cache_breakpoint": True},
        ]

        result = llm._format_messages_for_provider(messages)

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Test message"
        assert "cache_breakpoint" not in result[0]
