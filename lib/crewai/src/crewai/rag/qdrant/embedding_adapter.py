"""Adapter that lets CrewAI embedding functions be used by the Qdrant client."""

from typing import Any

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction


class QdrantEmbeddingAdapter:
    """Expose a CrewAI ``EmbeddingFunction`` through Qdrant's single-text protocol.

    CrewAI embedders take a list of texts and return a list of vectors, while
    Qdrant's ``EmbeddingFunction`` takes one string and returns one vector.

    Attributes:
        vector_size: Dimension of the vectors the wrapped function produces. It
            is measured on first use (one embedding call, unless text was already
            embedded) and then cached, so collections can be created with the
            embedder's real dimension.
    """

    def __init__(self, embedding_function: EmbeddingFunction[Any]) -> None:
        """Initialize the adapter.

        Args:
            embedding_function: CrewAI embedding function to wrap.
        """
        self._embedding_function = embedding_function
        self._vector_size: int | None = None

    def __call__(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        vector: list[float] = self._embedding_function([text])[0].tolist()
        self._vector_size = len(vector)
        return vector

    @property
    def vector_size(self) -> int:
        """Dimension of the produced vectors, measured lazily."""
        if self._vector_size is None:
            self._vector_size = len(self("dimension probe"))
        return self._vector_size
