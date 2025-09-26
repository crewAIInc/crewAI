"""Base embeddings callable utilities for RAG systems."""

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

from crewai.rag.core.types import (
    Embeddable,
    Embedding,
    Embeddings,
    PyEmbedding,
)

T = TypeVar("T")
D = TypeVar("D", bound=Embeddable, contravariant=True)


def normalize_embeddings(
    target: Embedding | list[Embedding] | PyEmbedding | list[PyEmbedding],
) -> Embeddings | None:
    """Normalize various embedding formats to a standard list of numpy arrays.

    Args:
        target: Input embeddings in various formats (list of floats, list of lists,
                numpy array, or list of numpy arrays).

    Returns:
        Normalized embeddings as a list of numpy arrays, or None if input is None.

    Raises:
        ValueError: If embeddings are empty or in an unsupported format.
    """
    if isinstance(target, np.ndarray):
        if target.ndim == 1:
            return [target.astype(np.float32)]
        if target.ndim == 2:
            return [row.astype(np.float32) for row in target]
        raise ValueError(f"Unsupported numpy array shape: {target.shape}")

    first = target[0]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return [np.array(target, dtype=np.float32)]
    if isinstance(first, list):
        return [np.array(emb, dtype=np.float32) for emb in target]
    if isinstance(first, np.ndarray):
        return [emb.astype(np.float32) for emb in target]  # type: ignore[union-attr]

    raise ValueError(f"Unsupported embeddings format: {type(first)}")


def maybe_cast_one_to_many(target: T | list[T] | None) -> list[T] | None:
    """Cast a single item to a list if needed.

    Args:
        target: A single item or list of items.

    Returns:
        A list of items or None if input is None.
    """
    if target is None:
        return None
    return target if isinstance(target, list) else [target]


def validate_embeddings(embeddings: Embeddings) -> Embeddings:
    """Validate embeddings format and content.

    Args:
        embeddings: List of numpy arrays to validate.

    Returns:
        Validated embeddings.

    Raises:
        ValueError: If embeddings format or content is invalid.
    """
    if not isinstance(embeddings, list):
        raise ValueError(
            f"Expected embeddings to be a list, got {type(embeddings).__name__}"
        )
    if len(embeddings) == 0:
        raise ValueError(
            f"Expected embeddings to be a list with at least one item, got {len(embeddings)} embeddings"
        )
    if not all(isinstance(e, np.ndarray) for e in embeddings):
        raise ValueError(
            "Expected each embedding in the embeddings to be a numpy array"
        )
    for i, embedding in enumerate(embeddings):
        if embedding.ndim == 0:
            raise ValueError(
                f"Expected a 1-dimensional array, got a 0-dimensional array {embedding}"
            )
        if embedding.size == 0:
            raise ValueError(
                f"Expected each embedding to be a 1-dimensional numpy array with at least 1 value. "
                f"Got an array with no values at position {i}"
            )
        if not all(
            isinstance(value, (np.integer, float, np.floating))
            and not isinstance(value, bool)
            for value in embedding
        ):
            raise ValueError(
                f"Expected embedding to contain numeric values, got non-numeric values at position {i}"
            )
    return embeddings


@runtime_checkable
class EmbeddingFunction(Protocol[D]):
    """Protocol for embedding functions.

    Embedding functions convert input data (documents or images) into vector embeddings.
    """

    def __call__(self, input: D) -> Embeddings:
        """Convert input data to embeddings.

        Args:
            input: Input data to embed (documents or images).

        Returns:
            List of numpy arrays representing the embeddings.
        """
        ...

    def __init_subclass__(cls) -> None:
        """Wrap __call__ method to normalize and validate embeddings."""
        super().__init_subclass__()
        original_call = cls.__call__

        def wrapped_call(self: EmbeddingFunction[D], input: D) -> Embeddings:
            result = original_call(self, input)
            if result is None:
                raise ValueError("Embedding function returned None")
            normalized = normalize_embeddings(result)
            if normalized is None:
                raise ValueError("Normalization returned None for non-None input")
            return validate_embeddings(normalized)

        cls.__call__ = wrapped_call  # type: ignore[method-assign]

    def embed_query(self, input: D) -> Embeddings:
        """
        Get the embeddings for a query input.
        This method is optional, and if not implemented, the default behavior is to call __call__.
        """
        return self.__call__(input=input)
