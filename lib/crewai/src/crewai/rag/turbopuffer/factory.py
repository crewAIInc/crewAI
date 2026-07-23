"""Factory functions for creating turbopuffer clients from configuration."""

from crewai.rag.turbopuffer.client import TurbopufferClient
from crewai.rag.turbopuffer.config import TurbopufferConfig


def create_client(config: TurbopufferConfig) -> TurbopufferClient:
    """Create a turbopuffer client from configuration.

    Args:
        config: The turbopuffer configuration containing a pre-configured
            turbopuffer client instance.

    Returns:
        A configured TurbopufferClient instance.
    """
    return TurbopufferClient(
        client=config.client,
        embedding_function=config.embedding_function,
        default_limit=config.limit,
        default_score_threshold=config.score_threshold,
        default_batch_size=config.batch_size,
        distance_metric=config.distance_metric,
    )
