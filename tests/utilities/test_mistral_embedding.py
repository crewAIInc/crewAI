"""
Test cases for Mistral embedding functionality in CrewAI.

This module tests the Mistral embedding implementation and its integration
with the CrewAI embedding configurator, following the same patterns as
other embedding provider tests.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.mistral_embedding_function import MistralEmbeddingFunction
from crewai.rag.embeddings.providers.mistral.mistral_provider import MistralProvider


class TestMistralEmbeddingFunction:
    """Test cases for MistralEmbeddingFunction."""

    def test_initialization_with_api_key(self):
        """Test MistralEmbeddingFunction initialization with explicit API key."""
        embedding_func = MistralEmbeddingFunction(
            api_key="test_api_key",
            model_name="mistral-embed",
            base_url="https://api.mistral.ai/v1"
        )

        assert embedding_func.api_key == "test_api_key"
        assert embedding_func.model_name == "mistral-embed"
        assert embedding_func.base_url == "https://api.mistral.ai/v1"
        assert embedding_func.max_retries == 3
        assert embedding_func.timeout == 30

    def test_initialization_with_env_var(self):
        """Test MistralEmbeddingFunction initialization with environment variable."""
        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_env_key'}):
            embedding_func = MistralEmbeddingFunction()

            assert embedding_func.api_key == 'test_env_key'
            assert embedding_func.model_name == "mistral-embed"  # default

    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Mistral API key is required"):
                MistralEmbeddingFunction()

    def test_model_info(self):
        """Test get_model_info method."""
        embedding_func = MistralEmbeddingFunction(
            api_key="test_api_key",
            model_name="mistral-embed"
        )

        info = embedding_func.get_model_info()

        assert info["provider"] == "mistral"
        assert info["model"] == "mistral-embed"
        assert "base_url" in info

    @patch('requests.post')
    def test_successful_embedding_generation(self, mock_post):
        """Test successful embedding generation with mocked API response."""
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

        embedding_func = MistralEmbeddingFunction(api_key="test_api_key")
        test_inputs = ["Hello world", "Test document"]

        embeddings = embedding_func(test_inputs)

        # Verify API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        assert call_args[0][0] == "https://api.mistral.ai/v1/embeddings"
        assert call_args[1]["headers"]["Authorization"] == "Bearer test_api_key"
        assert call_args[1]["json"]["model"] == "mistral-embed"
        assert call_args[1]["json"]["input"] == test_inputs

        # Verify embeddings were returned
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 5
        assert len(embeddings[1]) == 5

    @patch('requests.post')
    def test_single_string_input(self, mock_post):
        """Test embedding generation with single string input."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response

        embedding_func = MistralEmbeddingFunction(api_key="test_api_key")
        embeddings = embedding_func("Single document")

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3

    @patch('requests.post')
    def test_empty_input(self, mock_post):
        """Test embedding generation with empty input."""
        embedding_func = MistralEmbeddingFunction(api_key="test_api_key")

        # Empty input should raise a ValueError due to ChromaDB validation
        with pytest.raises(
            ValueError, match="Expected Embeddings to be non-empty list or numpy array"
        ):
            embedding_func([])

        mock_post.assert_not_called()

    @patch('requests.post')
    def test_api_error_handling(self, mock_post):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response

        embedding_func = MistralEmbeddingFunction(api_key="test_api_key")

        with pytest.raises(RuntimeError, match="Failed to get embeddings from Mistral API"):
            embedding_func(["Test document"])

    @patch('requests.post')
    def test_retry_logic(self, mock_post):
        """Test retry logic for failed requests."""
        # Mock first two calls to fail, third to succeed
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        mock_post.side_effect = [mock_response_fail, mock_response_fail, mock_response_success]

        embedding_func = MistralEmbeddingFunction(
            api_key="test_api_key",
            max_retries=3
        )

        embeddings = embedding_func(["Test document"])

        # Should have made 3 calls (2 failures + 1 success)
        assert mock_post.call_count == 3
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3

    @patch('requests.post')
    def test_max_retries_exceeded(self, mock_post):
        """Test behavior when max retries are exceeded."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        embedding_func = MistralEmbeddingFunction(
            api_key="test_api_key",
            max_retries=2
        )

        with pytest.raises(RuntimeError, match="Failed to get embeddings from Mistral API after 2 attempts"):
            embedding_func(["Test document"])

        # Should have made 2 calls (max_retries)
        assert mock_post.call_count == 2


class TestEmbeddingFactory:
    """Test cases for embedding factory with Mistral support."""

    def setup_method(self):
        """Set up test environment before each test."""

    def test_mistral_provider_support(self):
        """Test that mistral provider is supported."""
        # Test that we can create a Mistral provider
        provider = MistralProvider()
        assert provider is not None

    def test_configure_mistral_basic(self):
        """Test basic Mistral configuration."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
            }
        }

        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_api_key'}):
            with patch('crewai.rag.embeddings.providers.mistral.mistral_provider.MistralProvider._create_embedding_function') as mock_create:
                mock_instance = MagicMock()
                mock_create.return_value = mock_instance

                result = build_embedder(embedder_config)

                # Verify _create_embedding_function was called
                mock_create.assert_called_once()

                # Verify the result is the mocked instance
                assert result == mock_instance

    def test_configure_mistral_with_explicit_api_key(self):
        """Test Mistral configuration with explicit API key."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "api_key": "explicit_api_key",
                "model": "mistral-embed-v1",
                "base_url": "https://custom.mistral.ai/v1",
                "max_retries": 5,
                "timeout": 60
            }
        }

        with patch('crewai.rag.embeddings.providers.mistral.mistral_provider.MistralProvider._create_embedding_function') as mock_create:
            mock_instance = MagicMock()
            mock_create.return_value = mock_instance

            result = build_embedder(embedder_config)

            # Verify _create_embedding_function was called
            mock_create.assert_called_once()

            # Verify the result is the mocked instance
            assert result == mock_instance

    def test_configure_mistral_with_model_name_parameter(self):
        """Test Mistral configuration with model_name parameter."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed-v1"
            }
        }

        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_api_key'}):
            with patch('crewai.rag.embeddings.providers.mistral.mistral_provider.MistralProvider._create_embedding_function') as mock_create:
                mock_instance = MagicMock()
                mock_create.return_value = mock_instance

                # Test with model_name parameter (as would be passed by factory)
                result = build_embedder({
                    "provider": "mistral",
                    "config": {
                        **embedder_config["config"],
                        "model": "mistral-embed-v1"
                    }
                })

                # Verify _create_embedding_function was called
                mock_create.assert_called_once()

                # Verify the result is the mocked instance
                assert result == mock_instance

    def test_configure_mistral_fallback_to_config_model(self):
        """Test that config model is used when model_name is None."""
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
            }
        }

        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_api_key'}):
            with patch('crewai.rag.embeddings.providers.mistral.mistral_provider.MistralProvider._create_embedding_function') as mock_create:
                mock_instance = MagicMock()
                mock_create.return_value = mock_instance

                # Test with model_name=None (fallback to config)
                result = build_embedder({
                    "provider": "mistral",
                    "config": embedder_config["config"]
                })

                # Verify _create_embedding_function was called
                mock_create.assert_called_once()

                # Verify the result is the mocked instance
                assert result == mock_instance


class TestIntegration:
    """Integration tests for Mistral embedding functionality."""

    @patch('requests.post')
    def test_end_to_end_embedding_flow(self, mock_post):
        """Test complete end-to-end embedding flow."""
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

        # Configure embedder
        # Use factory directly
        embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
            }
        }

        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_api_key'}):
            embedding_function = build_embedder(embedder_config)

            # Test embedding generation
            test_documents = ["Document 1", "Document 2"]
            embeddings = embedding_function(test_documents)

            # Verify results
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 5
            assert len(embeddings[1]) == 5

            # Verify API was called correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "embeddings" in call_args[0][0]
            assert call_args[1]["json"]["input"] == test_documents

    def test_agent_configuration_compatibility(self):
        """Test compatibility with agent configuration format."""
        # Simulate agent embedder configuration
        agent_embedder_config = {
            "provider": "mistral",
            "config": {
                "model": "mistral-embed"
            }
        }

        # Use factory directly

        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test_api_key'}):
            with patch('crewai.rag.embeddings.providers.mistral.mistral_provider.MistralProvider._create_embedding_function') as mock_create:
                mock_instance = MagicMock()
                mock_create.return_value = mock_instance

                # Test configuration (should not raise exceptions)
                result = build_embedder(agent_embedder_config)

                assert result == mock_instance
                mock_create.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_provider_error(self):
        """Test error when using unsupported provider."""
        # Use factory directly

        embedder_config = {
            "provider": "unsupported_provider",
            "config": {}
        }

        with pytest.raises(Exception, match="Unknown provider"):
            build_embedder(embedder_config)

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        embedder_config = {
            "provider": "mistral",
            "config": {}
        }

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Mistral API key is required"):
                build_embedder(embedder_config)


# Pytest fixtures for integration testing
@pytest.fixture
def mock_mistral_api_response():
    """Fixture providing mock Mistral API response."""
    return {
        "id": "test_embedding_id",
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "index": 0
            }
        ],
        "model": "mistral-embed",
        "usage": {
            "prompt_tokens": 2,
            "total_tokens": 2
        }
    }


@pytest.fixture
def test_embedder_config():
    """Fixture providing test embedder configuration."""
    return {
        "provider": "mistral",
        "config": {
            "model": "mistral-embed",
            "api_key": "test_api_key"
        }
    }


def test_mistral_embedding_with_fixtures(mock_mistral_api_response, test_embedder_config):
    """Test using pytest fixtures."""
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_mistral_api_response
        mock_post.return_value = mock_response

        # Use factory directly
        embedding_function = build_embedder(test_embedder_config)

        embeddings = embedding_function(["Test document"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 5
