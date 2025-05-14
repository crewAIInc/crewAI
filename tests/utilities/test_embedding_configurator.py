import os
import pytest
from unittest.mock import patch, MagicMock

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


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
