"""Tests for OpenRouterCompletion class.

This module tests the OpenRouter provider implementation, specifically:
- API key handling (explicit and env var fallback)
- Site URL/name for attribution (explicit and env var fallback)
- Base URL override behavior
- Client parameter configuration
- Context window size
- Header injection
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from crewai.llms.providers.openrouter.completion import OpenRouterCompletion


class TestOpenRouterCompletionInit:
    """Tests for OpenRouterCompletion.__init__"""

    def test_init_with_explicit_api_key(self) -> None:
        """Test initialization with explicit api_key parameter."""
        with patch.object(OpenRouterCompletion, "_get_client_params") as mock_params:
            mock_params.return_value = {"api_key": "test-key"}
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="explicit-openrouter-key",
                    )

        assert llm.api_key == "explicit-openrouter-key"

    def test_init_with_openrouter_api_key_env_var(self) -> None:
        """Test initialization falls back to OPENROUTER_API_KEY env var."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "env-openrouter-key"},
            clear=False,
        ):
            with patch.object(
                OpenRouterCompletion, "_get_client_params"
            ) as mock_params:
                mock_params.return_value = {"api_key": "env-openrouter-key"}
                with patch("crewai.llms.providers.openai.completion.OpenAI"):
                    with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                        llm = OpenRouterCompletion(
                            model="anthropic/claude-3.5-sonnet",
                        )

        assert llm.api_key == "env-openrouter-key"

    def test_init_with_explicit_site_url_and_site_name(self) -> None:
        """Test initialization with explicit site_url and site_name parameters."""
        with patch.object(OpenRouterCompletion, "_get_client_params") as mock_params:
            mock_params.return_value = {"api_key": "test-key"}
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        site_url="https://myapp.com",
                        site_name="My Application",
                    )

        assert llm.site_url == "https://myapp.com"
        assert llm.site_name == "My Application"

    def test_init_with_site_url_and_site_name_env_vars(self) -> None:
        """Test initialization falls back to env vars for site_url and site_name."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_SITE_URL": "https://env-app.com",
                "OPENROUTER_SITE_NAME": "Env Application",
            },
            clear=False,
        ):
            with patch.object(
                OpenRouterCompletion, "_get_client_params"
            ) as mock_params:
                mock_params.return_value = {"api_key": "test-key"}
                with patch("crewai.llms.providers.openai.completion.OpenAI"):
                    with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                        llm = OpenRouterCompletion(
                            model="anthropic/claude-3.5-sonnet",
                        )

        assert llm.site_url == "https://env-app.com"
        assert llm.site_name == "Env Application"

    def test_explicit_site_params_override_env_vars(self) -> None:
        """Test that explicit site_url/site_name override env vars."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_SITE_URL": "https://env-app.com",
                "OPENROUTER_SITE_NAME": "Env Application",
            },
            clear=False,
        ):
            with patch.object(
                OpenRouterCompletion, "_get_client_params"
            ) as mock_params:
                mock_params.return_value = {"api_key": "test-key"}
                with patch("crewai.llms.providers.openai.completion.OpenAI"):
                    with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                        llm = OpenRouterCompletion(
                            model="anthropic/claude-3.5-sonnet",
                            site_url="https://explicit-app.com",
                            site_name="Explicit Application",
                        )

        assert llm.site_url == "https://explicit-app.com"
        assert llm.site_name == "Explicit Application"

    def test_base_url_is_always_overridden(self) -> None:
        """Test that base_url is always set to OpenRouter endpoint."""
        with patch.object(OpenRouterCompletion, "_get_client_params") as mock_params:
            mock_params.return_value = {"api_key": "test-key"}
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        base_url="https://should-be-ignored.com",
                    )

        assert llm.base_url == "https://openrouter.ai/api/v1"

    def test_base_url_kwarg_is_popped_and_ignored(self) -> None:
        """Test that any base_url passed in kwargs is removed and ignored."""
        with patch.object(OpenRouterCompletion, "_get_client_params") as mock_params:
            mock_params.return_value = {"api_key": "test-key"}
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    # This should not raise even with base_url in kwargs
                    llm = OpenRouterCompletion(
                        model="openai/gpt-4",
                        api_key="test-key",
                        base_url="https://wrong-url.com/v1",
                    )

        assert llm.base_url == "https://openrouter.ai/api/v1"


class TestOpenRouterGetClientParams:
    """Tests for OpenRouterCompletion._get_client_params"""

    def test_get_client_params_returns_correct_params(self) -> None:
        """Test that _get_client_params returns expected parameters."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-api-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-api-key",
                        timeout=30.0,
                        max_retries=3,
                    )

        params = llm._get_client_params()

        assert params["api_key"] == "test-api-key"
        assert params["base_url"] == "https://openrouter.ai/api/v1"
        assert params["timeout"] == 30.0
        assert params["max_retries"] == 3

    def test_get_client_params_raises_without_api_key(self) -> None:
        """Test that _get_client_params raises ValueError when no API key is available."""
        # Clear any existing OPENROUTER_API_KEY from environment
        env_without_key = {
            k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"
        }

        with patch.dict(os.environ, env_without_key, clear=True):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key=None,
                    )
                    # Manually set api_key to None to simulate missing key
                    llm.api_key = None

            with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
                llm._get_client_params()

    def test_openai_api_key_is_never_used(self) -> None:
        """Test that OPENAI_API_KEY is never used even if set.

        This ensures OpenRouterCompletion doesn't accidentally inherit
        OpenAI's API key fallback behavior from the parent class.
        """
        # Set OPENAI_API_KEY but NOT OPENROUTER_API_KEY
        env_with_only_openai_key = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
        }
        env_with_only_openai_key["OPENAI_API_KEY"] = "sk-openai-should-not-be-used"

        with patch.dict(os.environ, env_with_only_openai_key, clear=True):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                    )

            # api_key should be None, NOT the OPENAI_API_KEY value
            assert llm.api_key is None

            # _get_client_params should raise, not silently use OPENAI_API_KEY
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
                llm._get_client_params()

    def test_get_client_params_uses_env_var_fallback(self) -> None:
        """Test that _get_client_params falls back to OPENROUTER_API_KEY env var."""
        with patch("crewai.llms.providers.openai.completion.OpenAI"):
            with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                llm = OpenRouterCompletion(
                    model="anthropic/claude-3.5-sonnet",
                    api_key="initial-key",
                )
                # Simulate the key being cleared
                llm.api_key = None

        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "env-fallback-key"},
            clear=False,
        ):
            params = llm._get_client_params()
            assert params["api_key"] == "env-fallback-key"

    def test_get_client_params_filters_none_values(self) -> None:
        """Test that _get_client_params filters out None values."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        # Don't pass organization, project, etc.
                    )

        params = llm._get_client_params()

        # None values should be filtered out
        assert "organization" not in params
        assert "project" not in params
        assert "default_query" not in params


class TestOpenRouterContextWindow:
    """Tests for OpenRouterCompletion.get_context_window_size"""

    def test_get_context_window_size_returns_expected_value(self) -> None:
        """Test that get_context_window_size returns the expected default value."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                    )

        # Import the ratio to calculate expected value
        from crewai.llm import CONTEXT_WINDOW_USAGE_RATIO

        expected = int(128000 * CONTEXT_WINDOW_USAGE_RATIO)
        assert llm.get_context_window_size() == expected

    def test_context_window_size_is_positive(self) -> None:
        """Test that context window size is always positive."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="openai/gpt-4",
                        api_key="test-key",
                    )

        assert llm.get_context_window_size() > 0


class TestOpenRouterHeaderInjection:
    """Tests for OpenRouter-specific header injection"""

    def test_headers_injected_when_site_url_provided(self) -> None:
        """Test that HTTP-Referer header is set when site_url is provided."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        site_url="https://myapp.com",
                    )

        assert llm.default_headers is not None
        assert llm.default_headers.get("HTTP-Referer") == "https://myapp.com"

    def test_headers_injected_when_site_name_provided(self) -> None:
        """Test that X-Title header is set when site_name is provided."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        site_name="My Application",
                    )

        assert llm.default_headers is not None
        assert llm.default_headers.get("X-Title") == "My Application"

    def test_both_headers_injected_when_both_provided(self) -> None:
        """Test that both headers are set when both site_url and site_name are provided."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        site_url="https://myapp.com",
                        site_name="My Application",
                    )

        assert llm.default_headers is not None
        assert llm.default_headers.get("HTTP-Referer") == "https://myapp.com"
        assert llm.default_headers.get("X-Title") == "My Application"

    def test_no_headers_when_no_site_values(self) -> None:
        """Test that no special headers are added when site values are not provided."""
        # Ensure env vars are not set
        env_without_site = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OPENROUTER_SITE_URL", "OPENROUTER_SITE_NAME")
        }
        env_without_site["OPENROUTER_API_KEY"] = "test-key"

        with patch.dict(os.environ, env_without_site, clear=True):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                    )

        # default_headers should be None when no site values are provided
        assert llm.default_headers is None

    def test_headers_from_env_vars(self) -> None:
        """Test that headers are set from environment variables."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_SITE_URL": "https://env-app.com",
                "OPENROUTER_SITE_NAME": "Env App",
            },
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                    )

        assert llm.default_headers is not None
        assert llm.default_headers.get("HTTP-Referer") == "https://env-app.com"
        assert llm.default_headers.get("X-Title") == "Env App"

    def test_existing_default_headers_are_preserved(self) -> None:
        """Test that existing default_headers are preserved when adding OpenRouter headers."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        site_url="https://myapp.com",
                        default_headers={"Custom-Header": "custom-value"},
                    )

        assert llm.default_headers is not None
        assert llm.default_headers.get("Custom-Header") == "custom-value"
        assert llm.default_headers.get("HTTP-Referer") == "https://myapp.com"


class TestOpenRouterProviderIntegration:
    """Integration-style tests for OpenRouter provider configuration"""

    def test_provider_is_set_correctly(self) -> None:
        """Test that the provider is set correctly."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                    )

        assert llm.provider == "openrouter"

    def test_model_is_preserved(self) -> None:
        """Test that the model name is preserved correctly."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                    )

        assert llm.model == "anthropic/claude-3.5-sonnet"

    def test_inherits_from_openai_completion(self) -> None:
        """Test that OpenRouterCompletion inherits from OpenAICompletion."""
        from crewai.llms.providers.openai.completion import OpenAICompletion

        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                    )

        assert isinstance(llm, OpenAICompletion)

    def test_temperature_is_passed_through(self) -> None:
        """Test that temperature parameter is passed through correctly."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        temperature=0.7,
                    )

        assert llm.temperature == 0.7

    def test_max_tokens_is_passed_through(self) -> None:
        """Test that max_tokens parameter is passed through correctly."""
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key"},
            clear=False,
        ):
            with patch("crewai.llms.providers.openai.completion.OpenAI"):
                with patch("crewai.llms.providers.openai.completion.AsyncOpenAI"):
                    llm = OpenRouterCompletion(
                        model="anthropic/claude-3.5-sonnet",
                        api_key="test-key",
                        max_tokens=4096,
                    )

        assert llm.max_tokens == 4096
