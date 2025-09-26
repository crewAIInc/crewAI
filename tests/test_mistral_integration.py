"""
Integration test for Mistral embedding functionality.

This test verifies that Mistral embeddings work correctly with CrewAI's
knowledge management system, following the same patterns as other
embedding provider integration tests.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai.rag.embeddings.factory import build_embedder


class TestMistralIntegration:
    """Integration tests for Mistral embedding functionality."""

    def setup_method(self):
        """Set up test environment before each test."""

    @patch("requests.post")
    def test_mistral_embedding_with_knowledge_base(self, mock_post):
        """Test Mistral embeddings with CrewAI knowledge base."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
                {"embedding": [0.6, 0.7, 0.8, 0.9, 1.0]},
            ]
        }
        mock_post.return_value = mock_response

        # Configure Mistral embedder
        embedder_config = {"provider": "mistral", "config": {"model": "mistral-embed"}}

        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test_api_key"}):
            embedding_function = build_embedder(embedder_config)

            # Test with sample documents
            documents = [
                "This is a test document about artificial intelligence.",
                "Another document about machine learning and neural networks.",
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
        # Test that mistral provider is supported
        # Test that we can build a Mistral embedder
        embedder = build_embedder({"provider": "mistral", "config": {}})
        assert embedder is not None

    def test_mistral_configuration_parameters(self):
        """Test that Mistral configuration accepts all expected parameters."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "api_key": "test_api_key",
                "model": "mistral-embed",
                "base_url": "https://api.mistral.ai/v1",
                "max_retries": 5,
                "timeout": 60,
            },
        }

        with patch(
            "crewai.rag.embeddings.providers.mistral.mistral_provider.MistralProvider._create_embedding_function"
        ) as mock_create:
            mock_instance = MagicMock()
            mock_create.return_value = mock_instance

            build_embedder(embedder_config)

            # Verify _create_embedding_function was called
            mock_create.assert_called_once()

    @patch("requests.post")
    def test_mistral_error_handling(self, mock_post):
        """Test Mistral error handling and retry logic."""
        # Mock API error
        mock_response = MagicMock()
        mock_response.status_code = 429  # Rate limit
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "429 Rate Limit"
        )
        mock_post.return_value = mock_response

        embedder_config = {
            "provider": "mistral",
            "config": {"model": "mistral-embed", "max_retries": 2},
        }

        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test_api_key"}):
            embedding_function = build_embedder(embedder_config)

            with pytest.raises(
                RuntimeError, match="Failed to get embeddings from Mistral API"
            ):
                embedding_function(["Test document"])

            # Should have made 3 calls (1 initial + 2 retries)
            assert mock_post.call_count == 3

    def test_mistral_environment_variable_fallback(self):
        """Test that Mistral uses environment variable when API key not provided."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
                # No explicit api_key
            },
        }

        with patch.dict(os.environ, {"MISTRAL_API_KEY": "env_api_key"}):
            with patch(
                "crewai.rag.embeddings.providers.mistral.mistral_provider.MistralProvider._create_embedding_function"
            ) as mock_create:
                mock_instance = MagicMock()
                mock_create.return_value = mock_instance

                build_embedder(embedder_config)

                # Verify _create_embedding_function was called
                mock_create.assert_called_once()

    def test_mistral_missing_api_key_error(self):
        """Test error when API key is missing."""
        embedder_config = {"provider": "mistral", "config": {"model": "mistral-embed"}}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Mistral API key is required"):
                build_embedder(embedder_config)


# Pytest fixtures
@pytest.fixture
def mistral_embedder_config():
    """Fixture providing Mistral embedder configuration."""
    return {
        "provider": "mistral",
        "config": {"model": "mistral-embed", "api_key": "test_api_key"},
    }


@pytest.fixture
def mock_mistral_response():
    """Fixture providing mock Mistral API response."""
    return {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
            {"embedding": [0.6, 0.7, 0.8, 0.9, 1.0]},
        ]
    }


def test_mistral_embedding_with_fixtures(
    mistral_embedder_config, mock_mistral_response
):
    """Test Mistral embedding using pytest fixtures."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_mistral_response
        mock_post.return_value = mock_response

        # Use factory directly
        embedding_function = build_embedder(mistral_embedder_config)

        documents = ["Document 1", "Document 2"]
        embeddings = embedding_function(documents)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 5
        assert len(embeddings[1]) == 5


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
