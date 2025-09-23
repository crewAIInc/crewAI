import logging
import traceback
import warnings
from typing import Any, cast

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.rag.chromadb.types import ChromaEmbeddingFunctionWrapper
from crewai.rag.config.utils import get_rag_client
from crewai.rag.core.base_client import BaseClient
from crewai.rag.embeddings.factory import EmbedderConfig, get_embedding_function
from crewai.rag.factory import create_client
from crewai.rag.types import BaseRecord, SearchResult
from crewai.utilities.logger import Logger


class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    def __init__(
        self,
        embedder: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._client: BaseClient | None = None
        self._embedder_config = embedder  # Store embedder config

        warnings.filterwarnings(
            "ignore",
            message=r".*'model_fields'.*is deprecated.*",
            module=r"^chromadb(\.|$)",
        )

        if embedder:
            # Cast to EmbedderConfig for type checking
            embedder_typed = cast(EmbedderConfig, embedder) if embedder else None
            embedding_function = get_embedding_function(embedder_typed)
            batch_size = None
            if isinstance(embedder, dict) and "config" in embedder:
                nested_config = embedder["config"]
                if isinstance(nested_config, dict):
                    batch_size = nested_config.get("batch_size")

            # Create config with batch_size if provided
            if batch_size is not None:
                config = ChromaDBConfig(
                    embedding_function=cast(
                        ChromaEmbeddingFunctionWrapper, embedding_function
                    ),
                    batch_size=batch_size,
                )
            else:
                config = ChromaDBConfig(
                    embedding_function=cast(
                        ChromaEmbeddingFunctionWrapper, embedding_function
                    )
                )
            self._client = create_client(config)

    def _get_client(self) -> BaseClient:
        """Get the appropriate client - instance-specific or global."""
        return self._client if self._client else get_rag_client()

    def search(
        self,
        query: list[str],
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.6,
    ) -> list[SearchResult]:
        try:
            if not query:
                raise ValueError("Query cannot be empty")

            client = self._get_client()
            collection_name = (
                f"knowledge_{self.collection_name}"
                if self.collection_name
                else "knowledge"
            )
            query_text = " ".join(query) if len(query) > 1 else query[0]

            return client.search(
                collection_name=collection_name,
                query=query_text,
                limit=limit,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logging.error(
                f"Error during knowledge search: {e!s}\n{traceback.format_exc()}"
            )
            return []

    def reset(self) -> None:
        try:
            client = self._get_client()
            collection_name = (
                f"knowledge_{self.collection_name}"
                if self.collection_name
                else "knowledge"
            )
            client.delete_collection(collection_name=collection_name)
        except Exception as e:
            logging.error(
                f"Error during knowledge reset: {e!s}\n{traceback.format_exc()}"
            )

    def save(self, documents: list[str]) -> None:
        try:
            client = self._get_client()
            collection_name = (
                f"knowledge_{self.collection_name}"
                if self.collection_name
                else "knowledge"
            )
            client.get_or_create_collection(collection_name=collection_name)

            rag_documents: list[BaseRecord] = [{"content": doc} for doc in documents]

            batch_size = None
            if self._embedder_config and isinstance(self._embedder_config, dict):
                if "config" in self._embedder_config:
                    nested_config = self._embedder_config["config"]
                    if isinstance(nested_config, dict):
                        batch_size = nested_config.get("batch_size")

            if batch_size is not None:
                client.add_documents(
                    collection_name=collection_name,
                    documents=rag_documents,
                    batch_size=batch_size,
                )
            else:
                client.add_documents(
                    collection_name=collection_name, documents=rag_documents
                )
        except Exception as e:
            if "dimension mismatch" in str(e).lower():
                Logger(verbose=True).log(
                    "error",
                    "Embedding dimension mismatch. This usually happens when mixing different embedding models. Try resetting the collection using `crewai reset-memories -a`",
                    "red",
                )
                raise ValueError(
                    "Embedding dimension mismatch. Make sure you're using the same embedding model "
                    "across all operations with this collection."
                    "Try resetting the collection using `crewai reset-memories -a`"
                ) from e
            Logger(verbose=True).log("error", f"Failed to upsert documents: {e}", "red")
            raise
