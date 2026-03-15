"""OCI embedding provider exports."""

from crewai.rag.embeddings.providers.oci.embedding_callable import (
    OCIEmbeddingFunction,
)
from crewai.rag.embeddings.providers.oci.oci_provider import OCIProvider
from crewai.rag.embeddings.providers.oci.types import (
    OCIProviderConfig,
    OCIProviderSpec,
)


__all__ = [
    "OCIEmbeddingFunction",
    "OCIProvider",
    "OCIProviderConfig",
    "OCIProviderSpec",
]
