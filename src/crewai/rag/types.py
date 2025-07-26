"""Type definitions for RAG (Retrieval-Augmented Generation) systems."""

from collections.abc import Callable
from typing import TypeAlias

from pydantic import BaseModel, Field

from crewai.rag.core.base_document import BaseDocument



# Vector types
DenseVector: TypeAlias = list[float]
IntVector: TypeAlias = list[int]

# ChromaDB embedding types
# Excluding numpy array types for Pydantic compatibility
ChromaDbSingleEmbedding: TypeAlias = DenseVector | IntVector
ChromaDbEmbedding: TypeAlias = ChromaDbSingleEmbedding | list[DenseVector | IntVector]

# Qdrant embedding types
class QdrantSparseVector(BaseModel, extra="forbid"):
  """
  Sparse vector structure
  """

  indices: list[int] = Field(..., description="Indices must be unique")
  values: list[float] = Field(..., description="Values and indices must be the same length")

QdrantMultiVector: TypeAlias = list[DenseVector]
QdrantNamedVectors: TypeAlias = dict[str, DenseVector | QdrantSparseVector | QdrantMultiVector]
QdrantEmbedding: TypeAlias = DenseVector | QdrantMultiVector | QdrantNamedVectors | QdrantSparseVector

# Combined embedding types
Embedding: TypeAlias = ChromaDbEmbedding | QdrantEmbedding
EmbeddingFunction: TypeAlias = Callable[[list[BaseDocument]], list[Embedding]]
