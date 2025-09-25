"""Test Azure embedder configuration with nested format only."""

from unittest.mock import MagicMock, patch

from crewai.rag.embeddings.configurator import EmbeddingConfigurator


class TestAzureEmbedderConfiguration:
    """Test Azure embedder configuration with nested format."""

    @patch(
        "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction"
    )
    def test_azure_openai_with_nested_config(self, mock_openai_func):
        """Test Azure configuration using OpenAI provider with nested config key."""
        mock_embedding = MagicMock()
        mock_openai_func.return_value = mock_embedding

        configurator = EmbeddingConfigurator()

        embedder_config = {
            "provider": "openai",
            "config": {
                "api_key": "test-azure-key",
                "api_base": "https://test.openai.azure.com/",
                "api_type": "azure",
                "api_version": "2023-05-15",
                "model": "text-embedding-3-small",
                "deployment_id": "test-deployment",
            },
        }

        result = configurator.configure_embedder(embedder_config)

        mock_openai_func.assert_called_once_with(
            api_key="test-azure-key",
            model_name="text-embedding-3-small",
            api_base="https://test.openai.azure.com/",
            api_type="azure",
            api_version="2023-05-15",
            default_headers=None,
            dimensions=None,
            deployment_id="test-deployment",
            organization_id=None,
        )
        assert result == mock_embedding

    @patch(
        "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction"
    )
    def test_azure_provider_with_nested_config(self, mock_openai_func):
        """Test using 'azure' as provider with nested config."""
        mock_embedding = MagicMock()
        mock_openai_func.return_value = mock_embedding

        configurator = EmbeddingConfigurator()

        embedder_config = {
            "provider": "azure",
            "config": {
                "api_key": "test-azure-key",
                "api_base": "https://test.openai.azure.com/",
                "api_version": "2023-05-15",
                "model": "text-embedding-3-small",
                "deployment_id": "test-deployment",
            },
        }

        result = configurator.configure_embedder(embedder_config)

        mock_openai_func.assert_called_once_with(
            api_key="test-azure-key",
            api_base="https://test.openai.azure.com/",
            api_type="azure",
            api_version="2023-05-15",
            model_name="text-embedding-3-small",
            default_headers=None,
            dimensions=None,
            deployment_id="test-deployment",
            organization_id=None,
        )
        assert result == mock_embedding
