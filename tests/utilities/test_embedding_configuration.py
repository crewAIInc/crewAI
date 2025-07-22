from unittest.mock import patch

import pytest

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


def test_configure_embedder_importerror():
    configurator = EmbeddingConfigurator()
    
    embedder_config = {
        'provider': 'openai',
        'config': {
            'model': 'text-embedding-ada-002',
        }
    }
    
    with patch('chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction') as mock_openai:
        mock_openai.side_effect = ImportError("Module not found.")
        
        with pytest.raises(ImportError) as exc_info:
            configurator.configure_embedder(embedder_config)

        assert str(exc_info.value) == "Module not found."
        mock_openai.assert_called_once()
