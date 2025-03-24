import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


def test_openrouter_embedder_configuration():
    # Setup
    configurator = EmbeddingConfigurator()
    mock_openai_embedding = MagicMock()
    
    with patch(
        "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction",
        return_value=mock_openai_embedding,
    ) as mock_embedder:
        # Test with provided config
        embedder_config = {
            "provider": "openrouter",
            "config": {
                "api_key": "test-key",
                "model": "test-model",
            },
        }
        
        # Execute
        result = configurator.configure_embedder(embedder_config)
        
        # Verify
        assert result == mock_openai_embedding
        mock_embedder.assert_called_once_with(
            api_key="test-key",
            api_base="https://openrouter.ai/api/v1",
            model_name="test-model",
        )


def test_openrouter_embedder_configuration_with_env_var():
    # Setup
    configurator = EmbeddingConfigurator()
    mock_openai_embedding = MagicMock()
    
    # Test with API key from environment variable
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}), \
        patch(
            "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction",
            return_value=mock_openai_embedding,
        ) as mock_embedder:
        # Config without API key
        embedder_config = {
            "provider": "openrouter",
            "config": {
                "model": "test-model",
            },
        }
        
        # Execute
        result = configurator.configure_embedder(embedder_config)
        
        # Verify
        assert result == mock_openai_embedding
        mock_embedder.assert_called_once_with(
            api_key="env-key",
            api_base="https://openrouter.ai/api/v1",
            model_name="test-model",
        )


def test_openrouter_embedder_configuration_missing_api_key():
    # Setup
    configurator = EmbeddingConfigurator()
    
    # Test without API key
    with patch.dict(os.environ, {}, clear=True), \
        patch(
            "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction",
            side_effect=Exception("Should not be called"),
        ):
        # Config without API key
        embedder_config = {
            "provider": "openrouter",
            "config": {
                "model": "test-model",
            },
        }
        
        # Verify error is raised
        with pytest.raises(ValueError, match="OpenRouter API key must be provided"):
            configurator.configure_embedder(embedder_config)
