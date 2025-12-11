"""Tests for embedding function factory."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.rag.embeddings.factory import build_embedder


class TestEmbeddingFactory:
    """Test embedding factory functions."""

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_openai(self, mock_import):
        """Test building OpenAI embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "openai",
            "config": {
                "api_key": "test-key",
                "model_name": "text-embedding-3-small",
            },
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.openai.openai_provider.OpenAIProvider"
        )
        mock_provider_class.assert_called_once()

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["model_name"] == "text-embedding-3-small"

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_azure(self, mock_import):
        """Test building Azure embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "azure",
            "config": {
                "api_key": "test-azure-key",
                "api_base": "https://test.openai.azure.com/",
                "api_type": "azure",
                "api_version": "2023-05-15",
                "model_name": "text-embedding-3-small",
                "deployment_id": "test-deployment",
            },
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.microsoft.azure.AzureProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "test-azure-key"
        assert call_kwargs["api_base"] == "https://test.openai.azure.com/"
        assert call_kwargs["api_type"] == "azure"

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_ollama(self, mock_import):
        """Test building Ollama embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "ollama",
            "config": {
                "model_name": "nomic-embed-text",
                "url": "http://localhost:11434",
            },
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.ollama.ollama_provider.OllamaProvider"
        )

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_huggingface(self, mock_import):
        """Test building HuggingFace embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "huggingface",
            "config": {
                "api_key": "hf-test-key",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.huggingface.huggingface_provider.HuggingFaceProvider"
        )
        mock_provider_class.assert_called_once()

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["api_key"] == "hf-test-key"
        assert call_kwargs["model"] == "sentence-transformers/all-MiniLM-L6-v2"

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_cohere(self, mock_import):
        """Test building Cohere embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "cohere",
            "config": {
                "api_key": "cohere-key",
                "model_name": "embed-english-v3.0",
            },
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.cohere.cohere_provider.CohereProvider"
        )

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_voyageai(self, mock_import):
        """Test building VoyageAI embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "voyageai",
            "config": {
                "api_key": "voyage-key",
                "model": "voyage-2",
            },
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.voyageai.voyageai_provider.VoyageAIProvider"
        )

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_watsonx(self, mock_import):
        """Test building WatsonX embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "watsonx",
            "config": {
                "model_id": "ibm/slate-125m-english-rtrvr",
                "api_key": "watsonx-key",
                "url": "https://us-south.ml.cloud.ibm.com",
                "project_id": "test-project",
            },
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.ibm.watsonx.WatsonXProvider"
        )

    def test_build_embedder_unknown_provider(self):
        """Test error handling for unknown provider."""
        config = {"provider": "unknown-provider", "config": {}}

        with pytest.raises(ValueError, match="Unknown provider: unknown-provider"):
            build_embedder(config)

    def test_build_embedder_missing_provider(self):
        """Test error handling for missing provider key."""
        config = {"config": {"api_key": "test-key"}}

        with pytest.raises(KeyError):
            build_embedder(config)

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_import_error(self, mock_import):
        """Test error handling when provider import fails."""
        mock_import.side_effect = ImportError("Module not found")

        config = {"provider": "openai", "config": {"api_key": "test-key"}}

        with pytest.raises(ImportError, match="Failed to import provider openai"):
            build_embedder(config)

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_custom_provider(self, mock_import):
        """Test building custom embedder."""
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_callable = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable = mock_embedding_callable

        config = {
            "provider": "custom",
            "config": {"embedding_callable": mock_embedding_callable},
        }

        build_embedder(config)

        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.custom.custom_provider.CustomProvider"
        )

        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["embedding_callable"] == mock_embedding_callable

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    @patch("crewai.rag.embeddings.factory.build_embedder_from_provider")
    def test_build_embedder_with_provider_instance(
        self, mock_build_from_provider, mock_import
    ):
        """Test building embedder from provider instance."""
        from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider

        mock_provider = MagicMock(spec=BaseEmbeddingsProvider)
        mock_embedding_function = MagicMock()
        mock_build_from_provider.return_value = mock_embedding_function

        result = build_embedder(mock_provider)

        mock_build_from_provider.assert_called_once_with(mock_provider)
        assert result == mock_embedding_function
        mock_import.assert_not_called()
