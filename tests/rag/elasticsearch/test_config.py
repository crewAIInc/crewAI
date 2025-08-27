"""Tests for Elasticsearch configuration."""

import pytest

from crewai.rag.elasticsearch.config import ElasticsearchConfig


def test_elasticsearch_config_defaults():
    """Test that ElasticsearchConfig has correct defaults."""
    config = ElasticsearchConfig()
    
    assert config.provider == "elasticsearch"
    assert config.vector_dimension == 384
    assert config.similarity == "cosine"
    assert config.embedding_function is not None
    assert config.options["hosts"] == ["http://localhost:9200"]
    assert config.options["use_ssl"] is False


def test_elasticsearch_config_custom_options():
    """Test that ElasticsearchConfig accepts custom options."""
    custom_options = {
        "hosts": ["https://elastic.example.com:9200"],
        "username": "user",
        "password": "pass",
        "use_ssl": True,
    }
    
    config = ElasticsearchConfig(
        options=custom_options,
        vector_dimension=768,
        similarity="dot_product"
    )
    
    assert config.provider == "elasticsearch"
    assert config.vector_dimension == 768
    assert config.similarity == "dot_product"
    assert config.options["hosts"] == ["https://elastic.example.com:9200"]
    assert config.options["username"] == "user"
    assert config.options["use_ssl"] is True


def test_elasticsearch_config_embedding_function():
    """Test that embedding function works correctly."""
    config = ElasticsearchConfig()
    
    embedding = config.embedding_function("test text")
    
    assert isinstance(embedding, list)
    assert len(embedding) == config.vector_dimension
    assert all(isinstance(x, float) for x in embedding)
