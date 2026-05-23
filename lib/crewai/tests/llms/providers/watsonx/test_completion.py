"""Tests for IBM watsonx.ai provider."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest


class TestIAMTokenManager:
    """Tests for the IAM token manager."""

    def test_token_exchange_success(self):
        """Test successful IAM token exchange."""
        from crewai.llms.providers.watsonx.completion import _IAMTokenManager

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "test-bearer-token",
            "expiration": time.time() + 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            manager = _IAMTokenManager("test-api-key")
            token = manager.get_token()

            assert token == "test-bearer-token"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["data"]["apikey"] == "test-api-key"
            assert (
                call_kwargs[1]["data"]["grant_type"]
                == "urn:ibm:params:oauth:grant-type:apikey"
            )

    def test_token_caching(self):
        """Test that tokens are cached and not re-fetched."""
        from crewai.llms.providers.watsonx.completion import _IAMTokenManager

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "cached-token",
            "expiration": time.time() + 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            manager = _IAMTokenManager("test-api-key")

            # First call - should fetch
            token1 = manager.get_token()
            # Second call - should use cache
            token2 = manager.get_token()

            assert token1 == token2 == "cached-token"
            assert mock_post.call_count == 1  # Only one HTTP call

    def test_token_refresh_on_expiry(self):
        """Test that expired tokens are refreshed."""
        from crewai.llms.providers.watsonx.completion import _IAMTokenManager

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "access_token": f"token-{call_count}",
                "expiration": time.time() + (0 if call_count == 1 else 3600),
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.post", side_effect=mock_post):
            manager = _IAMTokenManager("test-api-key")

            # First call - gets token-1 which is already expired
            token1 = manager.get_token()
            assert token1 == "token-1"

            # Second call - token-1 is expired, should refresh to token-2
            token2 = manager.get_token()
            assert token2 == "token-2"
            assert call_count == 2

    def test_token_exchange_http_error(self):
        """Test that HTTP errors during token exchange raise RuntimeError."""
        from crewai.llms.providers.watsonx.completion import _IAMTokenManager

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )

        with patch("httpx.post", return_value=mock_response):
            manager = _IAMTokenManager("bad-api-key")
            with pytest.raises(RuntimeError, match="IBM IAM token exchange failed"):
                manager.get_token()


class TestWatsonxCompletionInit:
    """Tests for WatsonxCompletion initialization."""

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        with pytest.raises(ValueError, match="IBM Cloud API key is required"):
            WatsonxCompletion(model="ibm/granite-4-h-small")

    @patch.dict(
        os.environ,
        {"WATSONX_API_KEY": "test-key"},
        clear=True,
    )
    def test_missing_project_id_raises(self):
        """Test that missing project ID raises ValueError."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        with pytest.raises(ValueError, match="project ID is required"):
            WatsonxCompletion(model="ibm/granite-4-h-small")

    @patch.dict(
        os.environ,
        {
            "WATSONX_API_KEY": "test-key",
            "WATSONX_PROJECT_ID": "test-project",
        },
        clear=True,
    )
    def test_default_region_url(self):
        """Test that default region constructs correct URL."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "test-token",
            "expiration": time.time() + 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            with patch(
                "crewai.llms.providers.openai.completion.OpenAICompletion.__init__",
                return_value=None,
            ) as mock_init:
                completion = WatsonxCompletion.__new__(WatsonxCompletion)
                # Manually set _iam_manager and _project_id since we skip __init__
                # Instead, test the static method directly
                url = WatsonxCompletion._resolve_base_url(None, "us-south")
                assert url == "https://us-south.ml.cloud.ibm.com/ml/v1"

    def test_resolve_base_url_custom_region(self):
        """Test URL construction with custom region."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        url = WatsonxCompletion._resolve_base_url(None, "eu-de")
        assert url == "https://eu-de.ml.cloud.ibm.com/ml/v1"

    def test_resolve_base_url_explicit(self):
        """Test that explicit base_url takes priority."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        url = WatsonxCompletion._resolve_base_url(
            "https://custom.example.com/v1", "us-south"
        )
        assert url == "https://custom.example.com/v1"

    @patch.dict(
        os.environ,
        {"WATSONX_URL": "https://env-override.example.com/v1"},
        clear=True,
    )
    def test_resolve_base_url_env_override(self):
        """Test that WATSONX_URL env var overrides region-based URL."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        url = WatsonxCompletion._resolve_base_url(None, "us-south")
        assert url == "https://env-override.example.com/v1"

    def test_resolve_region_default(self):
        """Test default region resolution."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        with patch.dict(os.environ, {}, clear=True):
            region = WatsonxCompletion._resolve_region(None)
            assert region == "us-south"

    @patch.dict(os.environ, {"WATSONX_REGION": "eu-gb"}, clear=True)
    def test_resolve_region_from_env(self):
        """Test region resolution from environment variable."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        region = WatsonxCompletion._resolve_region(None)
        assert region == "eu-gb"

    def test_resolve_region_explicit(self):
        """Test explicit region parameter takes priority."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        region = WatsonxCompletion._resolve_region("jp-tok")
        assert region == "jp-tok"


class TestWatsonxModelCapabilities:
    """Tests for model capability detection."""

    def _make_completion(self, model: str) -> object:
        """Create a minimal WatsonxCompletion-like object for testing."""
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        # Create a bare instance without calling __init__
        obj = object.__new__(WatsonxCompletion)
        obj.model = model
        return obj

    def test_granite_4_context_window(self):
        """Test Granite 4.x models report 128K context."""
        comp = self._make_completion("ibm/granite-4-h-small")
        assert comp.get_context_window_size() == 131072

    def test_granite_3_instruct_context_window(self):
        """Test Granite 3.x instruct models report 128K context."""
        comp = self._make_completion("ibm/granite-3-8b-instruct")
        assert comp.get_context_window_size() == 131072

    def test_granite_code_context_window(self):
        """Test Granite code models report 8K context."""
        comp = self._make_completion("ibm/granite-8b-code-instruct")
        assert comp.get_context_window_size() == 8192

    def test_granite_4_supports_function_calling(self):
        """Test Granite 4.x models support function calling."""
        comp = self._make_completion("ibm/granite-4-h-small")
        assert comp.supports_function_calling() is True

    def test_granite_3_instruct_supports_function_calling(self):
        """Test Granite 3.x instruct models support function calling."""
        comp = self._make_completion("ibm/granite-3-8b-instruct")
        assert comp.supports_function_calling() is True

    def test_granite_guardian_no_function_calling(self):
        """Test Granite Guardian models don't support function calling."""
        comp = self._make_completion("ibm/granite-guardian-3-8b")
        assert comp.supports_function_calling() is False

    def test_granite_not_multimodal(self):
        """Test Granite models are not multimodal."""
        comp = self._make_completion("ibm/granite-4-h-small")
        assert comp.supports_multimodal() is False


class TestWatsonxModelRouting:
    """Tests for model routing through the LLM factory."""

    def test_watsonx_models_in_constants(self):
        """Test that WATSONX_MODELS is properly defined."""
        from crewai.llms.constants import WATSONX_MODELS

        assert "ibm/granite-4-h-small" in WATSONX_MODELS
        assert "ibm/granite-3-8b-instruct" in WATSONX_MODELS
        assert "ibm/granite-guardian-3-8b" in WATSONX_MODELS
        assert len(WATSONX_MODELS) >= 10

    def test_watsonx_in_supported_providers(self):
        """Test that watsonx is in the supported native providers list."""
        from crewai.llm import SUPPORTED_NATIVE_PROVIDERS

        assert "watsonx" in SUPPORTED_NATIVE_PROVIDERS
        assert "ibm" in SUPPORTED_NATIVE_PROVIDERS

    def test_get_native_provider_watsonx(self):
        """Test that _get_native_provider returns WatsonxCompletion."""
        from crewai.llm import LLM
        from crewai.llms.providers.watsonx.completion import WatsonxCompletion

        assert LLM._get_native_provider("watsonx") is WatsonxCompletion
        assert LLM._get_native_provider("ibm") is WatsonxCompletion

    def test_infer_provider_from_watsonx_model(self):
        """Test that Granite models are inferred as watsonx provider."""
        from crewai.llm import LLM

        assert LLM._infer_provider_from_model("ibm/granite-4-h-small") == "watsonx"
