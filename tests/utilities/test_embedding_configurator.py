import pytest
from unittest.mock import MagicMock, patch

from crewai.utilities.embedding_configurator import EmbeddingConfigurator
from crewai.utilities.embedding_functions import FixedGoogleVertexEmbeddingFunction


class TestEmbeddingConfigurator:
    @pytest.fixture
    def embedding_configurator(self):
        return EmbeddingConfigurator()
    
    def test_configure_vertexai(self, embedding_configurator):
        with patch('crewai.utilities.embedding_functions.FixedGoogleVertexEmbeddingFunction') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            
            config = {
                "provider": "vertexai",
                "config": {
                    "api_key": "test-key",
                    "model": "test-model",
                    "project_id": "test-project",
                    "region": "test-region"
                }
            }
            
            result = embedding_configurator.configure_embedder(config)
            
            mock_class.assert_called_once_with(
                model_name="test-model",
                api_key="test-key",
                project_id="test-project",
                region="test-region"
            )
            assert result == mock_instance
