"""VoyageAI embedding function implementation."""

from typing import cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


CONTEXTUALIZED_CHUNK_SIZE = 32000


class VoyageAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for VoyageAI models."""

    def __init__(self, **kwargs: Unpack[VoyageAIProviderConfig]) -> None:
        """Initialize VoyageAI embedding function.

        Args:
            **kwargs: Configuration parameters for VoyageAI.
        """
        try:
            import voyageai

        except ImportError as e:
            raise ImportError(
                "voyageai is required for voyageai embeddings. "
                "Install it with: uv add voyageai"
            ) from e
        self._config = kwargs
        self._client = voyageai.Client(  # type: ignore[attr-defined]
            api_key=kwargs["api_key"],
            max_retries=kwargs.get("max_retries", 0),
            timeout=kwargs.get("timeout"),
        )

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "voyageai"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors.
        """

        if isinstance(input, str):
            input = [input]

        model = self._config.get("model", "voyage-2")

        if model.startswith("voyage-context"):
            inputs = [[s] for s in input]
            ctx_result = self._client.contextualized_embed(
                inputs=inputs,
                model=model,
                input_type="document",
                output_dtype=self._config.get("output_dtype"),
                output_dimension=self._config.get("output_dimension"),
            )
            return cast(
                Embeddings,
                [r.embeddings[0] for r in ctx_result.results],
            )

        result = self._client.embed(
            texts=input,
            model=model,
            input_type=self._config.get("input_type"),
            truncation=self._config.get("truncation", True),
            output_dtype=self._config.get("output_dtype"),
            output_dimension=self._config.get("output_dimension"),
        )

        return cast(Embeddings, result.embeddings)
