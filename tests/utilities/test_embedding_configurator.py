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
