from unittest.mock import MagicMock, patch

import pytest

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


def test_azure_embedder_missing_deployment_id():
    """Test that Azure embedder raises an error when deployment_id is missing"""
    embedder_config = {
        "provider": "azure",
        "config": {
            "model": "text-embedding-ada-002",
            "api_key": "test-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
            "api_type": "azure",
        }
    }
    
    configurator = EmbeddingConfigurator()
    
    with pytest.raises(ValueError) as excinfo:
        configurator.configure_embedder(embedder_config)
        
    assert "Missing required parameter 'deployment_id'" in str(excinfo.value)


@patch("chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction")
def test_azure_embedder_with_deployment_id(mock_openai_embedding):
    """Test that Azure embedder works when deployment_id is provided"""
    mock_instance = MagicMock()
    mock_openai_embedding.return_value = mock_instance
    
    embedder_config = {
        "provider": "azure",
        "config": {
            "model": "text-embedding-ada-002",
            "api_key": "test-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
            "api_type": "azure",
            "deployment_id": "text-embedding-ada-002",
        }
    }
    
    configurator = EmbeddingConfigurator()
    result = configurator.configure_embedder(embedder_config)
    
    assert result == mock_instance
    mock_openai_embedding.assert_called_once()
    # Verify deployment_id was passed correctly
    call_kwargs = mock_openai_embedding.call_args.kwargs
    assert call_kwargs["deployment_id"] == "text-embedding-ada-002"
