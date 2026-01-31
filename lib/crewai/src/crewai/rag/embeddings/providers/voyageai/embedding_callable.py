"""VoyageAI embedding function implementation."""

from collections.abc import Callable, Generator
from typing import cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.voyageai.types import VoyageAIProviderConfig


# Token limits for different VoyageAI models
VOYAGE_TOTAL_TOKEN_LIMITS = {
    "voyage-context-3": 32_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-3.5": 320_000,
    "voyage-2": 320_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-large-2-instruct": 120_000,
    "voyage-finance-2": 120_000,
    "voyage-multilingual-2": 120_000,
    "voyage-law-2": 120_000,
    "voyage-large-2": 120_000,
    "voyage-3": 120_000,
    "voyage-3-lite": 120_000,
    "voyage-code-2": 120_000,
    "voyage-3-m-exp": 120_000,
    "voyage-multimodal-3": 32_000,
    "voyage-multimodal-3.5": 32_000,
    # Voyage-4 series models
    "voyage-4": 320_000,
    "voyage-4-lite": 1_000_000,
    "voyage-4-large": 120_000,
}

# Batch size for embedding requests
BATCH_SIZE = 1000


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

        # Use unified batching for all text inputs
        embeddings = self._embed_with_batching(list(input))

        return cast(Embeddings, embeddings)

    def _build_batches(self, texts: list[str]) -> Generator[list[str], None, None]:
        """
        Generate batches of texts based on token limits using a generator.

        Args:
            texts: List of texts to batch.

        Yields:
            Batches of texts as lists.
        """
        if not texts:
            return

        # Multimodal models use count-based batching (tokenize API doesn't support them)
        if self._is_multimodal_model():
            yield from self._build_batches_by_count(texts)
            return

        max_tokens_per_batch = self.get_token_limit()
        current_batch: list[str] = []
        current_batch_tokens = 0

        # Tokenize all texts in one API call
        all_token_lists = self._client.tokenize(texts, model=self._config["model"])
        token_counts = [len(tokens) for tokens in all_token_lists]

        for i, text in enumerate(texts):
            n_tokens = token_counts[i]

            # Check if adding this text would exceed limits
            if current_batch and (
                len(current_batch) >= BATCH_SIZE
                or (current_batch_tokens + n_tokens > max_tokens_per_batch)
            ):
                # Yield the current batch and start a new one
                yield current_batch
                current_batch = []
                current_batch_tokens = 0

            current_batch.append(text)
            current_batch_tokens += n_tokens

        # Yield the last batch (always has at least one text)
        if current_batch:
            yield current_batch

    def _build_batches_by_count(
        self, texts: list[str]
    ) -> Generator[list[str], None, None]:
        """
        Generate batches of texts based on count only (for multimodal models).

        Args:
            texts: List of texts to batch.

        Yields:
            Batches of texts as lists.
        """
        for i in range(0, len(texts), BATCH_SIZE):
            yield texts[i : i + BATCH_SIZE]

    def _get_embed_function(self) -> Callable[[list[str]], list[list[float]]]:
        """
        Get the appropriate embedding function based on model type.

        Returns:
            A callable that takes a batch of texts and returns embeddings.
        """
        model_name = self._config["model"]

        if self._is_context_model():

            def embed_batch_context(batch: list[str]) -> list[list[float]]:
                result = self._client.contextualized_embed(
                    inputs=[batch],
                    model=model_name,
                    input_type=self._config.get("input_type"),
                    output_dimension=self._config.get("output_dimension"),
                    output_dtype=self._config.get("output_dtype"),
                )
                return [list(emb) for emb in result.results[0].embeddings]

            return embed_batch_context

        if self._is_multimodal_model():

            def embed_batch_multimodal(batch: list[str]) -> list[list[float]]:
                # Multimodal API expects inputs as list of content lists
                # For text-only: [[text1], [text2], ...]
                inputs = [[text] for text in batch]
                result = self._client.multimodal_embed(
                    inputs=inputs,
                    model=model_name,
                    input_type=self._config.get("input_type"),
                    truncation=self._config.get("truncation", True),
                    # Note: output_dimension requires voyageai SDK >=0.3.6
                )
                return [list(emb) for emb in result.embeddings]

            return embed_batch_multimodal

        def embed_batch_regular(batch: list[str]) -> list[list[float]]:
            result = self._client.embed(
                texts=batch,
                model=model_name,
                input_type=self._config.get("input_type"),
                truncation=self._config.get("truncation", True),
                output_dimension=self._config.get("output_dimension"),
                output_dtype=self._config.get("output_dtype"),
            )
            return [list(emb) for emb in result.embeddings]

        return embed_batch_regular

    def _embed_with_batching(self, texts: list[str]) -> list[list[float]]:
        """
        Unified method to embed texts with automatic batching based on token limits.
        Works for regular and contextual models.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings.
        """
        if not texts:
            return []

        # Get the appropriate embedding function for this model type
        embed_fn = self._get_embed_function()

        # Process each batch
        all_embeddings = []
        for batch in self._build_batches(texts):
            batch_embeddings = embed_fn(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def count_tokens(self, texts: list[str]) -> list[int]:
        """
        Count tokens for the given texts.

        Args:
            texts: List of texts to count tokens for.

        Returns:
            List of token counts for each text.
        """
        if not texts:
            return []

        # Use the VoyageAI tokenize API to get token counts
        token_lists = self._client.tokenize(texts, model=self._config["model"])
        return [len(token_list) for token_list in token_lists]

    def get_token_limit(self) -> int:
        """
        Get the token limit for the current model.

        Returns:
            Token limit for the model, or default of 120_000 if not found.
        """
        model_name = self._config["model"]
        return VOYAGE_TOTAL_TOKEN_LIMITS.get(model_name, 120_000)

    def _is_context_model(self) -> bool:
        """Check if the model is a contextualized embedding model."""
        model_name = self._config["model"]
        return "context" in model_name

    def _is_multimodal_model(self) -> bool:
        """Check if the model is a multimodal embedding model."""
        model_name = self._config["model"]
        return "multimodal" in model_name
