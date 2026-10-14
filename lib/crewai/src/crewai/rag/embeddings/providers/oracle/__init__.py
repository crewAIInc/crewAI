"""Oracle embedding providers."""

from crewai.rag.embeddings.providers.oracle.embedding_callable import (
    OracleEmbeddingFunction,
)
from crewai.rag.embeddings.providers.oracle.oracle_provider import (
    OracleProvider,
)
from crewai.rag.embeddings.providers.oracle.types import (
    OracleProviderConfig,
    OracleProviderSpec,
)


__all__ = [
    "OracleEmbeddingFunction",
    "OracleProvider",
    "OracleProviderConfig",
    "OracleProviderSpec",
]
