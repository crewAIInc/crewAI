import os
from unittest.mock import MagicMock, patch

import pytest
from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
    AmazonBedrockEmbeddingFunction,
)

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


def test_embedding_bedrock_session():
    configurator = EmbeddingConfigurator()

    embedder_config = {
        'provider': 'bedrock',
        'config': {
            'model': 'bedrock_model',
            'session': 'session_name',
        }
    }

    with patch('chromadb.utils.embedding_functions.amazon_bedrock_embedding_function.AmazonBedrockEmbeddingFunction') as mock_bedrock:
        configurator.configure_embedder(embedder_config)

        mock_bedrock.assert_called_once_with(
            model_name = 'bedrock_model',session = 'session_name'
        )


def test_configure_bedrock_with_env_variables():
    configurator = EmbeddingConfigurator()

    mock_env = {
        "AWS_ACCESS_KEY_ID": "mock_access_key",
        "AWS_SECRET_ACCESS_KEY": "mock_secret_key",
        "AWS_REGION_NAME": "us-west-2"
    }
    
    mock_session = MagicMock()
    
    with patch.dict(os.environ, mock_env, clear=True), \
            patch('boto3.Session', return_value=mock_session) as mock_boto3_session, \
            patch('chromadb.utils.embedding_functions.amazon_bedrock_embedding_function.AmazonBedrockEmbeddingFunction') as mock_bedrock:
        
        embedder_config = {
            'provider': 'bedrock',
            'config': {
                'model': 'bedrock_model'
            }
        }
        
        configurator.configure_embedder(embedder_config)
        
        mock_boto3_session.assert_called_once_with(
            aws_access_key_id="mock_access_key",
            aws_secret_access_key="mock_secret_key",
            region_name="us-west-2"
        )

        mock_bedrock.assert_called_once_with(
            model_name='bedrock_model',
            session=mock_session
        )


def test_configure_bedrock_missing_env_variable():
    configurator = EmbeddingConfigurator()
    
    mock_env = {
        "AWS_ACCESS_KEY_ID": "mock_access_key",
        "AWS_REGION_NAME": "us-west-2"
    }
    
    with patch.dict(os.environ, mock_env, clear=True):
        embedder_config = {
            'provider': 'bedrock',
            'config': {
                'model': 'bedrock_model'
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            configurator.configure_embedder(embedder_config)
        
        assert "AWS_SECRET_ACCESS_KEY" in str(exc_info.value)
        assert "Missing required environment variables" in str(exc_info.value)
    


