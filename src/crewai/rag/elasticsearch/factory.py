"""Factory functions for creating Elasticsearch clients."""

from crewai.rag.elasticsearch.config import ElasticsearchConfig
from crewai.rag.elasticsearch.client import ElasticsearchClient


def create_client(config: ElasticsearchConfig) -> ElasticsearchClient:
    """Create an ElasticsearchClient from configuration.

    Args:
        config: Elasticsearch configuration object.

    Returns:
        Configured ElasticsearchClient instance.
    """
    try:
        from elasticsearch import Elasticsearch
    except ImportError as e:
        raise ImportError(
            "elasticsearch package is required for Elasticsearch support. "
            "Install it with: pip install elasticsearch"
        ) from e

    client = Elasticsearch(**config.options)
    
    return ElasticsearchClient(
        client=client,
        embedding_function=config.embedding_function,
        vector_dimension=config.vector_dimension,
        similarity=config.similarity,
    )
