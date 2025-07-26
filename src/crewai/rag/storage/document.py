from pydantic import Field

from crewai.rag.core.base_document import BaseDocument
from crewai.rag.types import Embedding


class VectorDocument(BaseDocument):
  """Document model extended with vector embedding for similarity search.

  This model extends the base Document with vector-specific fields needed
  for vector similarity search operations.

  Attributes:
      embedding: Vector representation of the document content. Supports various
          formats from different vector databases:
          - list[float]: Single dense vector (most common)
          - list[int]: Integer vectors
          - list[list[float]]: Multiple vectors
          - dict[str, ...]: Named vectors for multi-vector spaces
          - SparseVector: Sparse representation

  Notes:
    - Numpy arrays are supported at runtime and will be automatically
      converted to lists during serialization.
  """

  embedding: Embedding = Field(
    description="Vector embedding representation of the document"
  )
