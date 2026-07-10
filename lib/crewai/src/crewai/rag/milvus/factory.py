"""Factory functions for creating Milvus clients from configuration."""

from pymilvus import MilvusClient as SyncMilvusClient  # type: ignore[import-untyped]

from crewai.rag.milvus.client import MilvusClient
from crewai.rag.milvus.config import MilvusConfig


def create_client(config: MilvusConfig) -> MilvusClient:
    """Create a Milvus client from configuration.

    Args:
        config: The Milvus configuration.

    Returns:
        A configured MilvusClient instance.
    """

    milvus_client = SyncMilvusClient(**config.options)
    return MilvusClient(
        client=milvus_client,
        embedding_function=config.embedding_function,
        default_limit=config.limit,
        default_score_threshold=config.score_threshold,
        default_batch_size=config.batch_size,
        dimension=config.dimension,
        metric_type=config.metric_type,
        consistency_level=config.consistency_level,
    )
