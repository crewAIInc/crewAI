"""Custom embedding function base implementation."""

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.types import Documents, Embeddings


class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """Base class for custom embedding functions.

    This provides a concrete implementation that can be subclassed for custom embeddings.
    """

    def __call__(self, input: Documents) -> Embeddings:
        """Convert input documents to embeddings.

        Args:
            input: List of documents to embed.

        Returns:
            List of numpy arrays representing the embeddings.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")
