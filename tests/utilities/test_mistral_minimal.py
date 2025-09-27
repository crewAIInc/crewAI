"""
Minimal test for Mistral embedding functionality.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


class TestMistralMinimal:
    """Test cases for minimal Mistral embedding implementation."""

    def test_mistral_embedding_initialization(self):
        """Test that Mistral embedding function can be initialized."""
        from crewai.rag.embeddings.mistral_embedding_function import MistralEmbeddingFunction
        
        embedding_func = MistralEmbeddingFunction(
            api_key="test_api_key",
            model_name="mistral-embed"
        )
        
        assert embedding_func.api_key == "test_api_key"
        assert embedding_func.model_name == "mistral-embed"

    @patch("requests.post")
    def test_mistral_embedding_success(self, mock_post):
        """Test successful embedding generation."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
        mock_post.return_value = mock_response

        from crewai.rag.embeddings.mistral_embedding_function import MistralEmbeddingFunction
        
        embedding_func = MistralEmbeddingFunction(api_key="test_api_key")
        result = embedding_func(["Test document 1", "Test document 2"])
        
        assert len(result) == 2
        assert list(result[0]) == [0.1, 0.2, 0.3]
        assert list(result[1]) == [0.4, 0.5, 0.6]
        assert mock_post.call_count == 1

    def test_mistral_configurator_integration(self):
        """Test that Mistral can be configured through EmbeddingConfigurator."""
        configurator = EmbeddingConfigurator()
        
        embedder_config = {
            "provider": "mistral",
            "config": {
                "api_key": "test_api_key",
                "model": "mistral-embed"
            }
        }
        
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test_api_key"}):
            embedding_func = configurator.configure_embedder(embedder_config)
            assert embedding_func is not None
            assert embedding_func.model_name == "mistral-embed"

    def test_mistral_missing_api_key_error(self):
        """Test that missing API key raises appropriate error."""
        configurator = EmbeddingConfigurator()
        
        embedder_config = {
            "provider": "mistral",
            "config": {}
        }
        
        # Ensure no environment variable is set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Mistral API key is required"):
                configurator.configure_embedder(embedder_config)
