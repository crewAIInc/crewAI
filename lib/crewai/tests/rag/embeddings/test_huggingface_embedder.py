"""Tests for HuggingFace embedding function."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crewai.rag.embeddings.providers.huggingface.embedding_callable import (
    HuggingFaceEmbeddingFunction,
)


class TestHuggingFaceEmbeddingFunction:
    """Test HuggingFace embedding function."""

    @patch("huggingface_hub.InferenceClient")
    def test_initialization_with_api_key(self, mock_client_class):
        """Test initialization with API key."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        ef = HuggingFaceEmbeddingFunction(
            api_key="test-api-key",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        mock_client_class.assert_called_once_with(
            provider="hf-inference",
            token="test-api-key",
        )
        assert ef._model_name == "sentence-transformers/all-MiniLM-L6-v2"

    @patch("huggingface_hub.InferenceClient")
    def test_initialization_without_api_key(self, mock_client_class):
        """Test initialization without API key (for public models)."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        ef = HuggingFaceEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        mock_client_class.assert_called_once_with(
            provider="hf-inference",
            token=None,
        )
        assert ef._model_name == "sentence-transformers/all-MiniLM-L6-v2"

    @patch("huggingface_hub.InferenceClient")
    def test_initialization_with_default_model(self, mock_client_class):
        """Test initialization with default model name."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        ef = HuggingFaceEmbeddingFunction()

        assert ef._model_name == "sentence-transformers/all-MiniLM-L6-v2"

    @patch("huggingface_hub.InferenceClient")
    def test_call_with_single_document(self, mock_client_class):
        """Test embedding generation for a single document."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the feature_extraction response (1D embedding)
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.feature_extraction.return_value = mock_embedding

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")
        result = ef(["Hello, world!"])

        mock_client.feature_extraction.assert_called_once_with(
            text="Hello, world!",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        assert len(result) == 1
        assert result[0] == pytest.approx(mock_embedding, rel=1e-5)

    @patch("huggingface_hub.InferenceClient")
    def test_call_with_multiple_documents(self, mock_client_class):
        """Test embedding generation for multiple documents."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the feature_extraction response
        mock_embedding1 = [0.1, 0.2, 0.3]
        mock_embedding2 = [0.4, 0.5, 0.6]
        mock_client.feature_extraction.side_effect = [mock_embedding1, mock_embedding2]

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")
        result = ef(["Hello", "World"])

        assert mock_client.feature_extraction.call_count == 2
        assert len(result) == 2
        assert result[0] == pytest.approx(mock_embedding1, rel=1e-5)
        assert result[1] == pytest.approx(mock_embedding2, rel=1e-5)

    @patch("huggingface_hub.InferenceClient")
    def test_call_with_string_input(self, mock_client_class):
        """Test that string input is converted to list."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embedding = [0.1, 0.2, 0.3]
        mock_client.feature_extraction.return_value = mock_embedding

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")
        result = ef("Hello")  # type: ignore[arg-type]

        mock_client.feature_extraction.assert_called_once()
        assert len(result) == 1

    @patch("huggingface_hub.InferenceClient")
    def test_process_2d_embedding_result(self, mock_client_class):
        """Test processing of 2D token-level embeddings (mean pooling)."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock 2D token-level embeddings (3 tokens, 4 dimensions each)
        mock_token_embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ]
        mock_client.feature_extraction.return_value = mock_token_embeddings

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")
        result = ef(["Hello"])

        # Expected: mean pooling across tokens
        expected = np.mean(mock_token_embeddings, axis=0).tolist()
        assert len(result) == 1
        assert result[0] == pytest.approx(expected, rel=1e-5)

    @patch("huggingface_hub.InferenceClient")
    def test_process_3d_embedding_result(self, mock_client_class):
        """Test processing of 3D batch token-level embeddings."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock 3D embeddings (1 batch, 3 tokens, 4 dimensions)
        mock_batch_embeddings = [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ]
        ]
        mock_client.feature_extraction.return_value = mock_batch_embeddings

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")
        result = ef(["Hello"])

        # Expected: take first batch, then mean pooling
        expected = np.mean(mock_batch_embeddings[0], axis=0).tolist()
        assert len(result) == 1
        assert result[0] == pytest.approx(expected, rel=1e-5)

    @patch("huggingface_hub.InferenceClient")
    def test_error_handling_deprecated_endpoint(self, mock_client_class):
        """Test error handling for deprecated endpoint error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.feature_extraction.side_effect = Exception(
            "https://api-inference.huggingface.co is no longer supported"
        )

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")

        with pytest.raises(ValueError, match="HuggingFace API endpoint error"):
            ef(["Hello"])

    @patch("huggingface_hub.InferenceClient")
    def test_error_handling_unauthorized(self, mock_client_class):
        """Test error handling for authentication error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.feature_extraction.side_effect = Exception("401 Unauthorized")

        ef = HuggingFaceEmbeddingFunction(api_key="invalid-key")

        with pytest.raises(ValueError, match="HuggingFace API authentication error"):
            ef(["Hello"])

    @patch("huggingface_hub.InferenceClient")
    def test_error_handling_model_not_found(self, mock_client_class):
        """Test error handling for model not found error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.feature_extraction.side_effect = Exception("404 Not Found")

        ef = HuggingFaceEmbeddingFunction(
            api_key="test-key", model_name="nonexistent/model"
        )

        with pytest.raises(ValueError, match="HuggingFace model not found"):
            ef(["Hello"])

    @patch("huggingface_hub.InferenceClient")
    def test_error_handling_generic_error(self, mock_client_class):
        """Test error handling for generic API error."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_client.feature_extraction.side_effect = Exception("Some unexpected error")

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")

        with pytest.raises(ValueError, match="HuggingFace API error"):
            ef(["Hello"])

    @patch("huggingface_hub.InferenceClient")
    def test_name_method(self, mock_client_class):
        """Test the name() static method."""
        assert HuggingFaceEmbeddingFunction.name() == "huggingface"

    @patch("huggingface_hub.InferenceClient")
    def test_get_config(self, mock_client_class):
        """Test get_config method."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        ef = HuggingFaceEmbeddingFunction(
            api_key="test-key",
            model_name="custom/model",
        )

        config = ef.get_config()
        assert config["model_name"] == "custom/model"
        assert config["api_key"] == "test-key"


class TestHuggingFaceEmbeddingFunctionIntegration:
    """Integration tests for HuggingFace embedding function with RAGStorage."""

    @patch("huggingface_hub.InferenceClient")
    def test_embedding_function_works_with_rag_storage_validation(
        self, mock_client_class
    ):
        """Test that the embedding function works with RAGStorage validation.

        This test simulates the validation that happens in RAGStorage.__init__
        where it calls embedding_function(["test"]) to verify the embedder works.
        """
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock a valid embedding response
        mock_embedding = [0.1] * 384  # 384 dimensions like all-MiniLM-L6-v2
        mock_client.feature_extraction.return_value = mock_embedding

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")

        # This is what RAGStorage does to validate the embedder
        result = ef(["test"])

        assert len(result) == 1
        assert len(result[0]) == 384
        # Values should be numeric (float or numpy float)
        assert all(isinstance(x, (int, float)) or hasattr(x, "__float__") for x in result[0])

    @patch("huggingface_hub.InferenceClient")
    def test_embedding_function_returns_correct_format_for_chromadb(
        self, mock_client_class
    ):
        """Test that embeddings are in the correct format for ChromaDB.

        ChromaDB expects embeddings as a sequence of embedding vectors where each
        inner element is a 1D embedding vector with numeric values.
        """
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_client.feature_extraction.return_value = mock_embedding

        ef = HuggingFaceEmbeddingFunction(api_key="test-key")
        result = ef(["Hello", "World"])

        # ChromaDB expects a sequence of embedding vectors
        assert isinstance(result, list)
        for embedding in result:
            # Each embedding should be a sequence of numeric values
            assert len(embedding) == 5
            for value in embedding:
                # Values should be numeric (float or numpy float)
                assert isinstance(value, (int, float)) or hasattr(value, "__float__")
