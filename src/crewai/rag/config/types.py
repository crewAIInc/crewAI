"""Type definitions for RAG configuration."""

from typing import Annotated, TypeAlias, TYPE_CHECKING
from pydantic import Field

from crewai.rag.config.constants import DISCRIMINATOR

# Linter freaks out on conditional imports, assigning in the type checking fixes it
if TYPE_CHECKING:
    from crewai.rag.chromadb.config import ChromaDBConfig as ChromaDBConfig_

    ChromaDBConfig = ChromaDBConfig_
    from crewai.rag.qdrant.config import QdrantConfig as QdrantConfig_

    QdrantConfig = QdrantConfig_
    from crewai.rag.elasticsearch.config import ElasticsearchConfig as ElasticsearchConfig_

    ElasticsearchConfig = ElasticsearchConfig_
else:
    try:
        from crewai.rag.chromadb.config import ChromaDBConfig
    except ImportError:
        from crewai.rag.config.optional_imports.providers import (
            MissingChromaDBConfig as ChromaDBConfig,
        )

    try:
        from crewai.rag.qdrant.config import QdrantConfig
    except ImportError:
        from crewai.rag.config.optional_imports.providers import (
            MissingQdrantConfig as QdrantConfig,
        )

    try:
        from crewai.rag.elasticsearch.config import ElasticsearchConfig
    except ImportError:
        from crewai.rag.config.optional_imports.providers import (
            MissingElasticsearchConfig as ElasticsearchConfig,
        )

SupportedProviderConfig: TypeAlias = ChromaDBConfig | QdrantConfig | ElasticsearchConfig
RagConfigType: TypeAlias = Annotated[
    SupportedProviderConfig, Field(discriminator=DISCRIMINATOR)
]
