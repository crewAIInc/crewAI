"""Tests for Elasticsearch factory."""

import sys
from unittest.mock import Mock, patch

import pytest

from crewai.rag.elasticsearch.config import ElasticsearchConfig


def test_create_client():
    """Test that create_client creates an ElasticsearchClient."""
    config = ElasticsearchConfig()
    
    with patch.dict('sys.modules', {'elasticsearch': Mock()}):
        mock_elasticsearch_module = Mock()
        mock_client_instance = Mock()
        mock_elasticsearch_module.Elasticsearch.return_value = mock_client_instance
        
        with patch.dict('sys.modules', {'elasticsearch': mock_elasticsearch_module}):
            from crewai.rag.elasticsearch.factory import create_client
            client = create_client(config)
            
            mock_elasticsearch_module.Elasticsearch.assert_called_once_with(**config.options)
            assert client.client == mock_client_instance
            assert client.embedding_function == config.embedding_function
            assert client.vector_dimension == config.vector_dimension
            assert client.similarity == config.similarity


def test_create_client_missing_elasticsearch():
    """Test that create_client raises ImportError when elasticsearch is not installed."""
    config = ElasticsearchConfig()
    
    with patch.dict('sys.modules', {}, clear=False):
        if 'elasticsearch' in __import__('sys').modules:
            del __import__('sys').modules['elasticsearch']
        
        from crewai.rag.elasticsearch.factory import create_client
        with pytest.raises(ImportError, match="elasticsearch package is required"):
            create_client(config)
