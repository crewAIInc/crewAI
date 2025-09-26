"""VoyageAI embedding function implementation."""

from typing import cast

from typing_extensions import Unpack

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.types import Documents, Embeddings
from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for VoyageAI models."""

    def __init__(self, **kwargs: Unpack[VoyageAIProviderConfig]) -> None:
        """Initialize VoyageAI embedding function.

        Args:
            **kwargs: Configuration parameters for VoyageAI.
        """
        try:
            import voyageai  # type: ignore[import-not-found]

        except ImportError as e:
            raise ImportError(
                "voyageai is required for voyageai embeddings. "
                "Install it with: uv add voyageai"
            ) from e
        self._config = kwargs
        self._client = voyageai.Client(
            api_key=kwargs["api_key"],
            max_retries=kwargs.get("max_retries", 0),
            timeout=kwargs.get("timeout"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors.
        """

        if isinstance(input, str):
            input = [input]

        result = self._client.embed(
            texts=input,
            model=self._config.get("model", "voyage-2"),
            input_type=self._config.get("input_type"),
            truncation=self._config.get("truncation", True),
            output_dtype=self._config.get("output_dtype"),
            output_dimension=self._config.get("output_dimension"),
        )

        return cast(Embeddings, result.embeddings)
