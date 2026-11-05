"""Type definitions for RAG configuration."""

from typing import TYPE_CHECKING, TypeAlias

from crewai.rag.config.base import BaseRagConfig


if TYPE_CHECKING:
    from crewai.rag.chromadb.config import ChromaDBConfig
    from crewai.rag.qdrant.config import QdrantConfig

    SupportedProviderConfig: TypeAlias = ChromaDBConfig | QdrantConfig
    RagConfigType: TypeAlias = SupportedProviderConfig
else:
    # At runtime every provider config derives from BaseRagConfig. Naming the
    # concrete classes here would import every provider backend (e.g. chromadb)
    # whenever crewai is imported, even when it is not installed or not used.
    SupportedProviderConfig: TypeAlias = BaseRagConfig
    RagConfigType: TypeAlias = BaseRagConfig
