"""
Tests for the enhanced embedding service.
"""

import os
import pytest
from unittest.mock import Mock, patch

from crewai_tools.rag.embedding_service import EmbeddingService, EmbeddingConfig


class TestEmbeddingConfig:
    """Test the EmbeddingConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.batch_size == 100
        assert config.extra_config == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            provider="voyageai",
            model="voyage-2",
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
            batch_size=50,
            extra_config={"input_type": "document"}
        )

        assert config.provider == "voyageai"
        assert config.model == "voyage-2"
        assert config.api_key == "test-key"
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.batch_size == 50
        assert config.extra_config == {"input_type": "document"}


class TestEmbeddingService:
    """Test the EmbeddingService class."""

    def test_list_supported_providers(self):
        """Test listing supported providers."""
        providers = EmbeddingService.list_supported_providers()
        expected_providers = [
            "openai", "azure", "voyageai", "cohere", "google-generativeai",
            "amazon-bedrock", "huggingface", "jina", "ollama", "sentence-transformer",
            "instructor", "watsonx", "custom"
        ]

        assert isinstance(providers, list)
        assert len(providers) >= 15  # Should have at least 15 providers
        assert all(provider in providers for provider in expected_providers)

    def test_get_default_api_key(self):
        """Test getting default API keys from environment."""
        service = EmbeddingService.__new__(EmbeddingService)  # Create without __init__

        # Test with environment variable set
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            api_key = service._get_default_api_key("openai")
            assert api_key == "test-openai-key"

        # Test with no environment variable
        with patch.dict(os.environ, {}, clear=True):
            api_key = service._get_default_api_key("openai")
            assert api_key is None

        # Test unknown provider
        api_key = service._get_default_api_key("unknown-provider")
        assert api_key is None

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_initialization_success(self, mock_build_embedder):
        """Test successful initialization."""
        # Mock the embedding function
        mock_embedding_function = Mock()
        mock_build_embedder.return_value = mock_embedding_function

        service = EmbeddingService(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key"
        )

        assert service.config.provider == "openai"
        assert service.config.model == "text-embedding-3-small"
        assert service.config.api_key == "test-key"
        assert service._embedding_function == mock_embedding_function

        # Verify build_embedder was called with correct config
        mock_build_embedder.assert_called_once()
        call_args = mock_build_embedder.call_args[0][0]
        assert call_args["provider"] == "openai"
        assert call_args["config"]["api_key"] == "test-key"
        assert call_args["config"]["model_name"] == "text-embedding-3-small"

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_initialization_import_error(self, mock_build_embedder):
        """Test initialization with import error."""
        mock_build_embedder.side_effect = ImportError("CrewAI not installed")

        with pytest.raises(ImportError, match="CrewAI embedding providers not available"):
            EmbeddingService(provider="openai", model="test-model", api_key="test-key")

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_embed_text_success(self, mock_build_embedder):
        """Test successful text embedding."""
        # Mock the embedding function
        mock_embedding_function = Mock()
        mock_embedding_function.return_value = [[0.1, 0.2, 0.3]]
        mock_build_embedder.return_value = mock_embedding_function

        service = EmbeddingService(provider="openai", model="test-model", api_key="test-key")

        result = service.embed_text("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_embedding_function.assert_called_once_with(["test text"])

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_embed_text_empty_input(self, mock_build_embedder):
        """Test embedding empty text."""
        mock_embedding_function = Mock()
        mock_build_embedder.return_value = mock_embedding_function

        service = EmbeddingService(provider="openai", model="test-model", api_key="test-key")

        result = service.embed_text("")
        assert result == []

        result = service.embed_text("   ")
        assert result == []

        # Embedding function should not be called for empty text
        mock_embedding_function.assert_not_called()

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_embed_batch_success(self, mock_build_embedder):
        """Test successful batch embedding."""
        # Mock the embedding function
        mock_embedding_function = Mock()
        mock_embedding_function.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_build_embedder.return_value = mock_embedding_function

        service = EmbeddingService(provider="openai", model="test-model", api_key="test-key")

        texts = ["text1", "text2", "text3"]
        result = service.embed_batch(texts)

        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_embedding_function.assert_called_once_with(texts)

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_embed_batch_empty_input(self, mock_build_embedder):
        """Test batch embedding with empty input."""
        mock_embedding_function = Mock()
        mock_build_embedder.return_value = mock_embedding_function

        service = EmbeddingService(provider="openai", model="test-model", api_key="test-key")

        # Empty list
        result = service.embed_batch([])
        assert result == []

        # List with empty strings
        result = service.embed_batch(["", "   ", ""])
        assert result == []

        # Embedding function should not be called for empty input
        mock_embedding_function.assert_not_called()

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_validate_connection(self, mock_build_embedder):
        """Test connection validation."""
        # Mock successful embedding
        mock_embedding_function = Mock()
        mock_embedding_function.return_value = [[0.1, 0.2, 0.3]]
        mock_build_embedder.return_value = mock_embedding_function

        service = EmbeddingService(provider="openai", model="test-model", api_key="test-key")

        assert service.validate_connection() is True

        # Mock failed embedding
        mock_embedding_function.side_effect = Exception("Connection failed")
        assert service.validate_connection() is False

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_get_service_info(self, mock_build_embedder):
        """Test getting service information."""
        # Mock the embedding function
        mock_embedding_function = Mock()
        mock_embedding_function.return_value = [[0.1, 0.2, 0.3]]
        mock_build_embedder.return_value = mock_embedding_function

        service = EmbeddingService(provider="openai", model="test-model", api_key="test-key")

        info = service.get_service_info()

        assert info["provider"] == "openai"
        assert info["model"] == "test-model"
        assert info["embedding_dimension"] == 3
        assert info["batch_size"] == 100
        assert info["is_connected"] is True

    def test_create_openai_service(self):
        """Test OpenAI service creation."""
        with patch('crewai.rag.embeddings.factory.build_embedder'):
            service = EmbeddingService.create_openai_service(
                model="text-embedding-3-large",
                api_key="test-key"
            )

            assert service.config.provider == "openai"
            assert service.config.model == "text-embedding-3-large"
            assert service.config.api_key == "test-key"

    def test_create_voyage_service(self):
        """Test Voyage AI service creation."""
        with patch('crewai.rag.embeddings.factory.build_embedder'):
            service = EmbeddingService.create_voyage_service(
                model="voyage-large-2",
                api_key="test-key"
            )

            assert service.config.provider == "voyageai"
            assert service.config.model == "voyage-large-2"
            assert service.config.api_key == "test-key"

    def test_create_cohere_service(self):
        """Test Cohere service creation."""
        with patch('crewai.rag.embeddings.factory.build_embedder'):
            service = EmbeddingService.create_cohere_service(
                model="embed-multilingual-v3.0",
                api_key="test-key"
            )

            assert service.config.provider == "cohere"
            assert service.config.model == "embed-multilingual-v3.0"
            assert service.config.api_key == "test-key"

    def test_create_gemini_service(self):
        """Test Gemini service creation."""
        with patch('crewai.rag.embeddings.factory.build_embedder'):
            service = EmbeddingService.create_gemini_service(
                model="models/embedding-001",
                api_key="test-key"
            )

            assert service.config.provider == "google-generativeai"
            assert service.config.model == "models/embedding-001"
            assert service.config.api_key == "test-key"


class TestProviderConfigurations:
    """Test provider-specific configurations."""

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_openai_config(self, mock_build_embedder):
        """Test OpenAI configuration mapping."""
        mock_build_embedder.return_value = Mock()

        service = EmbeddingService(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            extra_config={"dimensions": 1024}
        )

        # Check the configuration passed to build_embedder
        call_args = mock_build_embedder.call_args[0][0]
        assert call_args["provider"] == "openai"
        assert call_args["config"]["api_key"] == "test-key"
        assert call_args["config"]["model_name"] == "text-embedding-3-small"
        assert call_args["config"]["dimensions"] == 1024

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_voyageai_config(self, mock_build_embedder):
        """Test Voyage AI configuration mapping."""
        mock_build_embedder.return_value = Mock()

        service = EmbeddingService(
            provider="voyageai",
            model="voyage-2",
            api_key="test-key",
            timeout=60.0,
            max_retries=5,
            extra_config={"input_type": "document"}
        )

        # Check the configuration passed to build_embedder
        call_args = mock_build_embedder.call_args[0][0]
        assert call_args["provider"] == "voyageai"
        assert call_args["config"]["api_key"] == "test-key"
        assert call_args["config"]["model"] == "voyage-2"
        assert call_args["config"]["timeout"] == 60.0
        assert call_args["config"]["max_retries"] == 5
        assert call_args["config"]["input_type"] == "document"

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_cohere_config(self, mock_build_embedder):
        """Test Cohere configuration mapping."""
        mock_build_embedder.return_value = Mock()

        service = EmbeddingService(
            provider="cohere",
            model="embed-english-v3.0",
            api_key="test-key"
        )

        # Check the configuration passed to build_embedder
        call_args = mock_build_embedder.call_args[0][0]
        assert call_args["provider"] == "cohere"
        assert call_args["config"]["api_key"] == "test-key"
        assert call_args["config"]["model_name"] == "embed-english-v3.0"

    @patch('crewai.rag.embeddings.factory.build_embedder')
    def test_gemini_config(self, mock_build_embedder):
        """Test Gemini configuration mapping."""
        mock_build_embedder.return_value = Mock()

        service = EmbeddingService(
            provider="google-generativeai",
            model="models/embedding-001",
            api_key="test-key"
        )

        # Check the configuration passed to build_embedder
        call_args = mock_build_embedder.call_args[0][0]
        assert call_args["provider"] == "google-generativeai"
        assert call_args["config"]["api_key"] == "test-key"
        assert call_args["config"]["model_name"] == "models/embedding-001"
