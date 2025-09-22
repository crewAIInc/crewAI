"""Test Azure embedder configuration with factory."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.rag.embeddings.factory import EmbedderConfig, get_embedding_function


class TestAzureEmbedderFactory:
    """Test Azure embedder configuration with factory function."""

    @patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS")
    def test_azure_with_nested_config(self, mock_providers):
        """Test Azure configuration with nested config key."""

        mock_embedding = MagicMock()
        mock_openai_func = MagicMock(return_value=mock_embedding)
        mock_providers.__getitem__.return_value = mock_openai_func
        mock_providers.__contains__.return_value = True

        embedder_config = EmbedderConfig(
            provider="openai",
            config={
                "api_key": "test-azure-key",
                "api_base": "https://test.openai.azure.com/",
                "api_type": "azure",
                "api_version": "2023-05-15",
                "model": "text-embedding-3-small",
                "deployment_id": "test-deployment",
            },
        )

        result = get_embedding_function(embedder_config)

        mock_openai_func.assert_called_once_with(
            api_key="test-azure-key",
            api_base="https://test.openai.azure.com/",
            api_type="azure",
            api_version="2023-05-15",
            model_name="text-embedding-3-small",
            deployment_id="test-deployment",
        )
        assert result == mock_embedding

    @patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS")
    def test_regular_openai_with_nested_config(self, mock_providers):
        """Test regular OpenAI configuration with nested config."""

        mock_embedding = MagicMock()
        mock_openai_func = MagicMock(return_value=mock_embedding)
        mock_providers.__getitem__.return_value = mock_openai_func
        mock_providers.__contains__.return_value = True

        embedder_config = EmbedderConfig(
            provider="openai",
            config={"api_key": "test-openai-key", "model": "text-embedding-3-large"},
        )

        result = get_embedding_function(embedder_config)

        mock_openai_func.assert_called_once_with(
            api_key="test-openai-key", model_name="text-embedding-3-large"
        )
        assert result == mock_embedding

    def test_flat_format_raises_error(self):
        """Test that flat format raises an error."""
        embedder_config = {
            "provider": "openai",
            "api_key": "test-key",
            "model_name": "text-embedding-3-small",
        }

        with pytest.raises(ValueError) as exc_info:
            get_embedding_function(embedder_config)

        assert "Invalid embedder configuration format" in str(exc_info.value)
        assert "nested under a 'config' key" in str(exc_info.value)
