"""Test DeepInfra embedder configuration with factory."""

from unittest.mock import MagicMock, patch

import pytest
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from pydantic import ValidationError

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.deepinfra.deepinfra_provider import (
    DeepInfraProvider,
)


class TestDeepInfraEmbedderFactory:
    """Test DeepInfra embedder configuration with factory function."""

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_deepinfra_with_nested_config(self, mock_import):
        """Test DeepInfra configuration with nested config key."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
            "provider": "deepinfra",
            "config": {
                "api_key": "test-deepinfra-key",
                "model_name": "Qwen/Qwen3-Embedding-4B",
                "api_base": "https://api.deepinfra.com/v1/openai",
                "default_headers": {"X-Test": "crewai"},
            },
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.deepinfra.deepinfra_provider.DeepInfraProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-deepinfra-key"
        assert call_kwargs["model_name"] == "Qwen/Qwen3-Embedding-4B"
        assert call_kwargs["api_base"] == "https://api.deepinfra.com/v1/openai"
        assert call_kwargs["default_headers"] == {"X-Test": "crewai"}

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_deepinfra_with_model_alias(self, mock_import):
        """Test DeepInfra configuration with 'model' alias instead of 'model_name'."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        embedder_config = {
            "provider": "deepinfra",
            "config": {
                "api_key": "test-deepinfra-key",
                "model": "intfloat/multilingual-e5-large",
            },
        }

        result = build_embedder(embedder_config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.deepinfra.deepinfra_provider.DeepInfraProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-deepinfra-key"
        assert call_kwargs["model"] == "intfloat/multilingual-e5-large"

        assert result == mock_embedding_function

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_deepinfra_import_error(self, mock_import):
        """Test handling of import errors for DeepInfra provider."""
        mock_import.side_effect = ImportError("Failed to import DeepInfra provider")

        embedder_config = {
            "provider": "deepinfra",
            "config": {"api_key": "test-key"},
        }

        with pytest.raises(ImportError) as exc_info:
            build_embedder(embedder_config)

        assert "Failed to import provider deepinfra" in str(exc_info.value)

    def test_model_alias_reaches_embedding_function_as_model_name(self):
        """Test the 'model' alias is forwarded as model_name, not as a stray kwarg."""
        embedder = build_embedder(
            {
                "provider": "deepinfra",
                "config": {"api_key": "test-key", "model": "Qwen/Qwen3-Embedding-4B"},
            }
        )

        assert isinstance(embedder, OpenAIEmbeddingFunction)
        assert embedder.model_name == "Qwen/Qwen3-Embedding-4B"


class TestDeepInfraProviderDirect:
    """Test DeepInfraProvider Pydantic settings model directly."""

    def test_default_values(self):
        """Test default values for DeepInfraProvider."""
        provider = DeepInfraProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model_name == "Qwen/Qwen3-Embedding-8B"
        assert provider.api_base == "https://api.deepinfra.com/v1/openai"
        assert provider.default_headers is None

    def test_custom_values(self):
        """Test custom configuration values."""
        provider = DeepInfraProvider(
            api_key="test-custom-key",
            model="Qwen/Qwen3-Embedding-4B",
            api_base="https://proxy.example.com/v1/openai",
            default_headers={"X-Test": "crewai"},
        )

        assert provider.api_key == "test-custom-key"
        assert provider.model_name == "Qwen/Qwen3-Embedding-4B"
        assert provider.api_base == "https://proxy.example.com/v1/openai"
        assert provider.default_headers == {"X-Test": "crewai"}

    def test_dimensions_is_rejected(self):
        """Test dimensions fails loudly instead of being dropped by the embedding function."""
        with pytest.raises(ValidationError, match="dimensions is not supported"):
            DeepInfraProvider(api_key="test-key", dimensions=1024)

    def test_dimensions_is_rejected_through_factory(self):
        """Test an embedder config with dimensions fails at build time with the reason."""
        with pytest.raises(ValidationError, match="text-embedding-3"):
            build_embedder(
                {
                    "provider": "deepinfra",
                    "config": {"api_key": "test-key", "dimensions": 1024},
                }
            )

    def test_dimensions_none_is_accepted(self):
        """Test an explicit None for dimensions is a no-op."""
        provider = DeepInfraProvider(api_key="test-key", dimensions=None)

        assert provider.model_name == "Qwen/Qwen3-Embedding-8B"

    def test_missing_api_key_raises_validation_error(self, monkeypatch):
        """Test that missing API key raises ValidationError when no env vars set."""
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)

        with pytest.raises(ValidationError):
            DeepInfraProvider()

    def test_env_var_deepinfra_api_key(self, monkeypatch):
        """Test resolving API key from DEEPINFRA_API_KEY env var."""
        monkeypatch.setenv("DEEPINFRA_API_KEY", "di-env-key")

        provider = DeepInfraProvider()
        assert provider.api_key == "di-env-key"

    def test_env_var_deepinfra_api_base(self, monkeypatch):
        """Test resolving the base URL from DEEPINFRA_API_BASE env var."""
        monkeypatch.setenv("DEEPINFRA_API_KEY", "di-env-key")
        monkeypatch.setenv("DEEPINFRA_API_BASE", "https://proxy.example.com/v1/openai")

        provider = DeepInfraProvider()
        assert provider.api_base == "https://proxy.example.com/v1/openai"

    def test_model_alias_normalization(self):
        """Test 'model' parameter maps to 'model_name'."""
        provider = DeepInfraProvider(
            api_key="test-key",
            model="intfloat/multilingual-e5-large",
        )
        assert provider.model_name == "intfloat/multilingual-e5-large"

    def test_model_name_takes_precedence(self):
        """Test that model_name takes precedence over model if both are given."""
        provider = DeepInfraProvider(
            api_key="test-key",
            model="BAAI/bge-m3",
            model_name="Qwen/Qwen3-Embedding-4B",
        )
        assert provider.model_name == "Qwen/Qwen3-Embedding-4B"
