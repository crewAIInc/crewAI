import unittest
from unittest.mock import patch, MagicMock

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


class TestEmbeddingConfigurator(unittest.TestCase):
    @patch("chromadb.utils.embedding_functions.amazon_bedrock_embedding_function.AmazonBedrockEmbeddingFunction")
    def test_configure_bedrock(self, mock_bedrock_embedder):
        """Test that the Bedrock embedder is configured correctly."""
        config = {"session": MagicMock()}
        model_name = "amazon.titan-embed-text-v1"
        
        embedder = EmbeddingConfigurator()._configure_bedrock(config, model_name)
        
        mock_bedrock_embedder.assert_called_once_with(
            session=config["session"],
            model_name=model_name,
            operation_name="InvokeModel",
        )
        self.assertEqual(embedder, mock_bedrock_embedder.return_value)
