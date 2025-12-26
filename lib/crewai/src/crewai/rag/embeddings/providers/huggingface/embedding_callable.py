"""HuggingFace embedding function implementation using huggingface_hub."""

from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import numpy as np
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.huggingface.types import HuggingFaceProviderConfig


class HuggingFaceEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for HuggingFace models using the Inference API.

    This implementation uses huggingface_hub's InferenceClient instead of the
    deprecated api-inference.huggingface.co endpoint that chromadb uses.
    """

    def __init__(self, **kwargs: Unpack[HuggingFaceProviderConfig]) -> None:
        """Initialize HuggingFace embedding function.

        Args:
            **kwargs: Configuration parameters for HuggingFace.
                - api_key: HuggingFace API key (optional for public models)
                - model_name: Model name to use for embeddings
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for HuggingFace embeddings. "
                "Install it with: uv add huggingface_hub"
            ) from e

        self._config = kwargs
        self._model_name = kwargs.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        api_key = kwargs.get("api_key")

        self._client = InferenceClient(
            provider="hf-inference",
            token=api_key,
        )

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "huggingface"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If the API returns an error or unexpected response format.
        """
        if isinstance(input, str):
            input = [input]

        embeddings: list[list[float]] = []

        for text in input:
            embedding = self._get_embedding_for_text(text)
            embeddings.append(embedding)

        return embeddings

    def _get_embedding_for_text(self, text: str) -> list[float]:
        """Get embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            ValueError: If the API returns an error.
        """
        try:
            result = self._client.feature_extraction(
                text=text,
                model=self._model_name,
            )

            # Handle different response formats
            return self._process_embedding_result(result)

        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages for common issues
            if "deprecated" in error_msg.lower() or "no longer supported" in error_msg.lower():
                raise ValueError(
                    f"HuggingFace API endpoint error: {error_msg}. "
                    "Please ensure you have the latest version of huggingface_hub installed."
                ) from e
            if "unauthorized" in error_msg.lower() or "401" in error_msg:
                raise ValueError(
                    f"HuggingFace API authentication error: {error_msg}. "
                    "Please check your API key configuration."
                ) from e
            if "not found" in error_msg.lower() or "404" in error_msg:
                raise ValueError(
                    f"HuggingFace model not found: {error_msg}. "
                    f"Please verify the model name '{self._model_name}' is correct "
                    "and supports feature extraction."
                ) from e
            raise ValueError(f"HuggingFace API error: {error_msg}") from e

    def _process_embedding_result(self, result: Any) -> list[float]:
        """Process the embedding result from the API.

        The HuggingFace API can return different formats depending on the model:
        - 1D array: Direct embedding vector
        - 2D array: Token-level embeddings (needs pooling)
        - Nested structure: Various model-specific formats

        Args:
            result: The raw result from the API.

        Returns:
            A 1D list of floats representing the embedding.

        Raises:
            ValueError: If the result format is unexpected.
        """
        # Convert to numpy array for easier processing
        arr = np.array(result)

        # Handle different dimensionalities
        if arr.ndim == 1:
            # Already a 1D embedding vector
            return arr.astype(np.float32).tolist()
        if arr.ndim == 2:
            # Token-level embeddings - apply mean pooling
            pooled = np.mean(arr, axis=0)
            return pooled.astype(np.float32).tolist()
        if arr.ndim == 3:
            # Batch of token-level embeddings - take first and apply mean pooling
            pooled = np.mean(arr[0], axis=0)
            return pooled.astype(np.float32).tolist()
        raise ValueError(
            f"Unexpected embedding result shape: {arr.shape}. "
            "Expected 1D, 2D, or 3D array."
        )

    def get_config(self) -> dict[str, Any]:
        """Return the configuration for serialization."""
        return {
            "model_name": self._model_name,
            "api_key": self._config.get("api_key"),
        }
