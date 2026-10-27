"""Oracle embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
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


def __getattr__(name: str) -> object:
    return load_provider(
        name,
        {
            "OracleEmbeddingFunction": f"{__name__}.embedding_callable",
            "OracleProvider": f"{__name__}.oracle_provider",
        },
    )
