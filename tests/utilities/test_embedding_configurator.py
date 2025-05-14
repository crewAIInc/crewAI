import pytest
from unittest.mock import patch, MagicMock

from crewai.utilities.embedding_configurator import EmbeddingConfigurator
from crewai.knowledge.embedder.chromadb.utils.embedding_functions.voyageai_embedding_function import VoyageAIEmbeddingFunction


class TestEmbeddingConfigurator:
    def test_configure_voyageai_embedder(self):
        """Test that the VoyageAI embedder is configured correctly."""
        with patch(
            "crewai.utilities.embedding_configurator.VoyageAIEmbeddingFunction"
        ) as mock_voyageai:
            mock_instance = MagicMock()
            mock_voyageai.return_value = mock_instance

            config = {"api_key": "test-key"}
            model_name = "voyage-3"

            configurator = EmbeddingConfigurator()
            embedder = configurator._configure_voyageai(config, model_name)

            mock_voyageai.assert_called_once_with(
                model_name=model_name, api_key="test-key"
            )
            assert embedder == mock_instance

    def test_configure_embedder_with_voyageai(self):
        """Test that the embedder configurator correctly handles VoyageAI provider."""
        with patch(
            "crewai.utilities.embedding_configurator.VoyageAIEmbeddingFunction"
        ) as mock_voyageai:
            mock_instance = MagicMock()
            mock_voyageai.return_value = mock_instance

            embedder_config = {
                "provider": "voyageai",
                "config": {"api_key": "test-key", "model": "voyage-3"},
            }

            configurator = EmbeddingConfigurator()
            embedder = configurator.configure_embedder(embedder_config)

            mock_voyageai.assert_called_once_with(
                model_name="voyage-3", api_key="test-key"
            )
            assert embedder == mock_instance
            
    def test_configure_voyageai_embedder_missing_api_key(self):
        """Test that the VoyageAI embedder raises an error when API key is missing."""
        with patch(
            "crewai.utilities.embedding_configurator.VoyageAIEmbeddingFunction"
        ) as mock_voyageai:
            mock_voyageai.side_effect = ValueError("API key is required for VoyageAI embeddings")

            config = {}  # Empty config without API key
            model_name = "voyage-3"

            configurator = EmbeddingConfigurator()
            
            with pytest.raises(ValueError, match="API key is required"):
                configurator._configure_voyageai(config, model_name)
                
    def test_configure_voyageai_embedder_custom_model(self):
        """Test that the VoyageAI embedder works with different model names."""
        with patch(
            "crewai.utilities.embedding_configurator.VoyageAIEmbeddingFunction"
        ) as mock_voyageai:
            mock_instance = MagicMock()
            mock_voyageai.return_value = mock_instance

            config = {"api_key": "test-key"}
            model_name = "voyage-3.5-lite"  # Using a different model

            configurator = EmbeddingConfigurator()
            embedder = configurator._configure_voyageai(config, model_name)

            mock_voyageai.assert_called_once_with(
                model_name=model_name, api_key="test-key"
            )
            assert embedder == mock_instance
