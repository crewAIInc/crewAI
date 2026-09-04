"""Test OpenRouter embedder configuration with factory."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.openrouter.openrouter_provider import (
    OpenRouterProvider,
)


class TestOpenRouterEmbedderFactory:
    """Test OpenRouter embedder configuration with factory function."""

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_openrouter_with_nested_config(self, mock_import):
        """Test OpenRouter configuration with nested config key."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
            "provider": "openrouter",
            "config": {
                "api_key": "test-openrouter-key",
                "model_name": "openai/text-embedding-3-large",
                "api_base": "https://openrouter.ai/api/v1",
                "dimensions": 1536,
            },
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.openrouter.openrouter_provider.OpenRouterProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-openrouter-key"
        assert call_kwargs["model_name"] == "openai/text-embedding-3-large"
        assert call_kwargs["api_base"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["dimensions"] == 1536

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_openrouter_with_model_alias(self, mock_import):
        """Test OpenRouter configuration with 'model' alias instead of 'model_name'."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
            "provider": "openrouter",
            "config": {
                "api_key": "test-openrouter-key",
                "model": "cohere/embed-multilingual-v3.0",
            },
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.openrouter.openrouter_provider.OpenRouterProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-openrouter-key"
        assert call_kwargs["model"] == "cohere/embed-multilingual-v3.0"

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_openrouter_import_error(self, mock_import):
        """Test handling of import errors for OpenRouter provider."""
        mock_import.side_effect = ImportError("Failed to import OpenRouter provider")

        embedder_config = {
            "provider": "openrouter",
            "config": {"api_key": "test-key"},
        }

        with pytest.raises(ImportError) as exc_info:
            build_embedder(embedder_config)

        assert "Failed to import provider openrouter" in str(exc_info.value)


class TestOpenRouterProviderDirect:
    """Test OpenRouterProvider Pydantic settings model directly."""

    def test_default_values(self):
        """Test default values for OpenRouterProvider."""
        provider = OpenRouterProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model_name == "openai/text-embedding-3-small"
        assert provider.api_base == "https://openrouter.ai/api/v1"
        assert provider.dimensions is None
        assert provider.organization_id is None
        assert provider.default_headers is None

    def test_custom_values(self):
        """Test custom configuration values."""
        provider = OpenRouterProvider(
            api_key="test-custom-key",
            model="openai/text-embedding-3-large",
            api_base="https://custom.openrouter.ai/api/v1",
            dimensions=3072,
            organization_id="org-123",
            default_headers={"HTTP-Referer": "https://crewai.com"},
        )

        assert provider.api_key == "test-custom-key"
        assert provider.model_name == "openai/text-embedding-3-large"
        assert provider.api_base == "https://custom.openrouter.ai/api/v1"
        assert provider.dimensions == 3072
        assert provider.organization_id == "org-123"
        assert provider.default_headers == {"HTTP-Referer": "https://crewai.com"}

    def test_missing_api_key_raises_validation_error(self, monkeypatch):
        """Test that missing API key raises ValidationError when no env vars set."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDINGS_OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValidationError):
            OpenRouterProvider()

    def test_env_var_openrouter_api_key(self, monkeypatch):
        """Test resolving API key from OPENROUTER_API_KEY env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        monkeypatch.delenv("EMBEDDINGS_OPENROUTER_API_KEY", raising=False)

        provider = OpenRouterProvider()
        assert provider.api_key == "sk-or-env-key"

    def test_env_var_embeddings_openrouter_api_key(self, monkeypatch):
        """Test resolving API key from EMBEDDINGS_OPENROUTER_API_KEY env var."""
        monkeypatch.setenv("EMBEDDINGS_OPENROUTER_API_KEY", "sk-or-embed-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        provider = OpenRouterProvider()
        assert provider.api_key == "sk-or-embed-key"

    def test_model_alias_normalization(self):
        """Test 'model' parameter maps to 'model_name'."""
        provider = OpenRouterProvider(
            api_key="test-key",
            model="cohere/embed-multilingual-v3.0",
        )
        assert provider.model_name == "cohere/embed-multilingual-v3.0"

    def test_model_name_takes_precedence(self):
        """Test that model_name takes precedence over model if both are given."""
        provider = OpenRouterProvider(
            api_key="test-key",
            model="openai/text-embedding-3-small",
            model_name="openai/text-embedding-3-large",
        )
        assert provider.model_name == "openai/text-embedding-3-large"
