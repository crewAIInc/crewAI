"""FastEmbed embedding function implementation."""

from typing import Any, cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.fastembed.types import FastEmbedProviderConfig


class FastEmbedEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for FastEmbed text embedding models."""

    def __init__(self, **kwargs: Unpack[FastEmbedProviderConfig]) -> None:
        """Initialize FastEmbed embedding function.

        Args:
            **kwargs: Configuration parameters for FastEmbed.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "fastembed is required for fastembed embeddings. "
                "Install it with: uv add fastembed"
            ) from e

        model_kwargs: dict[str, Any] = {
            "model_name": kwargs.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
        }
        for key in (
            "cache_dir",
            "threads",
            "providers",
            "cuda",
            "device_ids",
            "lazy_load",
        ):
            if key in kwargs and kwargs[key] is not None:
                model_kwargs[key] = kwargs[key]

        self._model = TextEmbedding(**model_kwargs)
        self._batch_size = kwargs.get("batch_size", 256)
        self._parallel = kwargs.get("parallel")

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "fastembed"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors.
        """
        if isinstance(input, str):
            input = [input]

        embed_kwargs: dict[str, Any] = {"batch_size": self._batch_size}
        if self._parallel is not None:
            embed_kwargs["parallel"] = self._parallel

        return cast(Embeddings, list(self._model.embed(input, **embed_kwargs)))
