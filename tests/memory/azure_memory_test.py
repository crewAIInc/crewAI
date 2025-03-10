from unittest.mock import MagicMock, patch

import pytest

from crewai.utilities.embedding_configurator import EmbeddingConfigurator

# Test constants for Azure OpenAI configurations
AZURE_BASE_CONFIG = {
    "provider": "azure",
    "config": {
        "model": "text-embedding-ada-002",
        "api_key": "test-key",
        "api_base": "https://test.openai.azure.com",
        "api_version": "2023-05-15",
        "api_type": "azure",
    }
}

AZURE_COMPLETE_CONFIG = {
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


def test_azure_embedder_missing_deployment_id():
    """Test that Azure embedder raises an error when deployment_id is missing"""
    embedder_config = AZURE_BASE_CONFIG.copy()
    
    configurator = EmbeddingConfigurator()
    
    with pytest.raises(ValueError) as excinfo:
        configurator.configure_embedder(embedder_config)
        
    assert "Missing required parameters" in str(excinfo.value)
    assert "deployment ID (deployment_id)" in str(excinfo.value)


def test_azure_embedder_missing_api_key():
    """Test that Azure embedder raises an error when api_key is missing"""
    embedder_config = AZURE_BASE_CONFIG.copy()
    embedder_config["config"] = embedder_config["config"].copy()
    embedder_config["config"]["deployment_id"] = "text-embedding-ada-002"
    embedder_config["config"].pop("api_key")
    
    configurator = EmbeddingConfigurator()
    
    with pytest.raises(ValueError) as excinfo:
        configurator.configure_embedder(embedder_config)
        
    assert "Missing required parameters" in str(excinfo.value)
    assert "API key (api_key)" in str(excinfo.value)


def test_azure_embedder_missing_api_base():
    """Test that Azure embedder raises an error when api_base is missing"""
    embedder_config = AZURE_BASE_CONFIG.copy()
    embedder_config["config"] = embedder_config["config"].copy()
    embedder_config["config"]["deployment_id"] = "text-embedding-ada-002"
    embedder_config["config"].pop("api_base")
    
    configurator = EmbeddingConfigurator()
    
    with pytest.raises(ValueError) as excinfo:
        configurator.configure_embedder(embedder_config)
        
    assert "Missing required parameters" in str(excinfo.value)
    assert "API base URL (api_base)" in str(excinfo.value)


def test_azure_embedder_missing_api_version():
    """Test that Azure embedder raises an error when api_version is missing"""
    embedder_config = AZURE_BASE_CONFIG.copy()
    embedder_config["config"] = embedder_config["config"].copy()
    embedder_config["config"]["deployment_id"] = "text-embedding-ada-002"
    embedder_config["config"].pop("api_version")
    
    configurator = EmbeddingConfigurator()
    
    with pytest.raises(ValueError) as excinfo:
        configurator.configure_embedder(embedder_config)
        
    assert "Missing required parameters" in str(excinfo.value)
    assert "API version (api_version)" in str(excinfo.value)


def test_azure_embedder_empty_parameters():
    """Test that Azure embedder raises an error when parameters are empty strings"""
    embedder_config = AZURE_BASE_CONFIG.copy()
    embedder_config["config"] = embedder_config["config"].copy()
    embedder_config["config"]["deployment_id"] = ""
    embedder_config["config"]["api_key"] = ""
    
    configurator = EmbeddingConfigurator()
    
    with pytest.raises(ValueError) as excinfo:
        configurator.configure_embedder(embedder_config)
        
    assert "Missing required parameters" in str(excinfo.value)
    assert "API key (api_key)" in str(excinfo.value)
    assert "deployment ID (deployment_id)" in str(excinfo.value)


@patch("chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction")
def test_azure_embedder_with_all_required_parameters(mock_openai_embedding):
    """Test that Azure embedder works when all required parameters are provided"""
    mock_instance = MagicMock()
    mock_openai_embedding.return_value = mock_instance
    
    embedder_config = AZURE_COMPLETE_CONFIG.copy()
    
    configurator = EmbeddingConfigurator()
    result = configurator.configure_embedder(embedder_config)
    
    assert result == mock_instance
    mock_openai_embedding.assert_called_once()
    # Verify parameters were passed correctly
    call_kwargs = mock_openai_embedding.call_args.kwargs
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["api_base"] == "https://test.openai.azure.com"
    assert call_kwargs["api_version"] == "2023-05-15"
    assert call_kwargs["deployment_id"] == "text-embedding-ada-002"
