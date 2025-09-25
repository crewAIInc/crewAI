"""
Integration test for Mistral embedding functionality.

This test verifies that Mistral embeddings work correctly with CrewAI's
knowledge management system, following the same patterns as other
embedding provider integration tests.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import requests

from crewai.rag.embeddings.configurator import EmbeddingConfigurator


class TestMistralIntegration:
    """Integration tests for Mistral embedding functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.configurator = EmbeddingConfigurator()
    
    @patch('requests.post')
    def test_mistral_embedding_with_knowledge_base(self, mock_post):
        """Test Mistral embeddings with CrewAI knowledge base."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
                {"embedding": [0.6, 0.7, 0.8, 0.9, 1.0]}
            ]
        }
        mock_post.return_value = mock_response
        
        # Configure Mistral embedder
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
            }
        }
        
        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_api_key'}):
            embedding_function = self.configurator.configure_embedder(embedder_config)
            
            # Test with sample documents
            documents = [
                "This is a test document about artificial intelligence.",
                "Another document about machine learning and neural networks."
            ]
            
            embeddings = embedding_function(documents)
            
            # Verify embeddings were generated
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 5
            assert len(embeddings[1]) == 5
            
            # Verify API was called correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "embeddings" in call_args[0][0]
            assert call_args[1]["json"]["input"] == documents
            assert call_args[1]["json"]["model"] == "mistral-embed"
    
    def test_mistral_provider_availability(self):
        """Test that Mistral provider is available in configurator."""
        assert "mistral" in self.configurator.embedding_functions
        assert hasattr(self.configurator, '_configure_mistral')
    
    def test_mistral_configuration_parameters(self):
        """Test that Mistral configuration accepts all expected parameters."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "api_key": "test_api_key",
                "model": "mistral-embed",
                "base_url": "https://api.mistral.ai/v1",
                "max_retries": 5,
                "timeout": 60
            }
        }
        
        with patch('crewai.rag.embeddings.mistral_embedding_function.MistralEmbeddingFunction') as mock_mistral:
            mock_instance = MagicMock()
            mock_mistral.return_value = mock_instance
            
            result = self.configurator.configure_embedder(embedder_config)
            
            # Verify all parameters were passed correctly
            call_kwargs = mock_mistral.call_args[1]
            assert call_kwargs['api_key'] == "test_api_key"
            assert call_kwargs['model_name'] == "mistral-embed"
            assert call_kwargs['base_url'] == "https://api.mistral.ai/v1"
            assert call_kwargs['max_retries'] == 5
            assert call_kwargs['timeout'] == 60
    
    @patch('requests.post')
    def test_mistral_error_handling(self, mock_post):
        """Test Mistral error handling and retry logic."""
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 429  # Rate limit
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Rate Limit")
        mock_post.return_value = mock_response
        
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed",
                "max_retries": 2
            }
        }
        
        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_api_key'}):
            embedding_function = self.configurator.configure_embedder(embedder_config)
            
            with pytest.raises(RuntimeError, match="Failed to get embeddings from Mistral API"):
                embedding_function(["Test document"])
            
            # Should have made 2 calls (max_retries)
            assert mock_post.call_count == 2
    
    def test_mistral_environment_variable_fallback(self):
        """Test that Mistral uses environment variable when API key not provided."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
                # No explicit api_key
            }
        }
        
        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'env_api_key'}):
            with patch('crewai.rag.embeddings.mistral_embedding_function.MistralEmbeddingFunction') as mock_mistral:
                mock_instance = MagicMock()
                mock_mistral.return_value = mock_instance
                
                result = self.configurator.configure_embedder(embedder_config)
                
                # Verify environment variable was used
                call_kwargs = mock_mistral.call_args[1]
                assert call_kwargs['api_key'] == 'env_api_key'
    
    def test_mistral_missing_api_key_error(self):
        """Test error when API key is missing."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
            }
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Mistral API key is required"):
                self.configurator.configure_embedder(embedder_config)


# Pytest fixtures
@pytest.fixture
def mistral_embedder_config():
    """Fixture providing Mistral embedder configuration."""
    return {
        "provider": "mistral",
        "config": {
            "model": "mistral-embed",
            "api_key": "test_api_key"
        }
    }


@pytest.fixture
def mock_mistral_response():
    """Fixture providing mock Mistral API response."""
    return {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
            {"embedding": [0.6, 0.7, 0.8, 0.9, 1.0]}
        ]
    }


def test_mistral_embedding_with_fixtures(mistral_embedder_config, mock_mistral_response):
    """Test Mistral embedding using pytest fixtures."""
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_mistral_response
        mock_post.return_value = mock_response
        
        configurator = EmbeddingConfigurator()
        embedding_function = configurator.configure_embedder(mistral_embedder_config)
        
        documents = ["Document 1", "Document 2"]
        embeddings = embedding_function(documents)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 5
        assert len(embeddings[1]) == 5


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])