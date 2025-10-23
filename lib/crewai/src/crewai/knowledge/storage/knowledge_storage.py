from collections.abc import Mapping, Sequence
import logging
import traceback
from typing import Any, cast
import warnings

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.rag.chromadb.types import ChromaEmbeddingFunctionWrapper
from crewai.rag.config.utils import get_rag_client
from crewai.rag.core.base_client import BaseClient
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.types import ProviderSpec
from crewai.rag.factory import create_client
from crewai.rag.types import BaseRecord, SearchResult
from crewai.utilities.logger import Logger


def _coerce_to_records(documents: Sequence[Any]) -> list[BaseRecord]:
    records: list[BaseRecord] = []
    for d in documents:
        if isinstance(d, str):
            records.append({"content": d})
        elif isinstance(d, Mapping):
            content = d.get("content")
            if not content:
                continue
            metadata_raw = d.get("metadata", {})
            metadata: Any = {}
            if isinstance(metadata_raw, Mapping):
                metadata = {str(k): v for k, v in metadata_raw.items()}
            elif isinstance(metadata_raw, list):
                metadata = [
                    {str(k): v for k, v in m.items()}
                    for m in metadata_raw
                    if isinstance(m, Mapping)
                ]
            else:
                metadata = {}

            rec: BaseRecord = {"content": cast(str, content)}
            if metadata:
                rec["metadata"] = metadata

            if "doc_id" in d and isinstance(d["doc_id"], str):
                rec["doc_id"] = d["doc_id"]
            records.append(rec)
        else:
            continue
    return records


class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    def __init__(
        self,
        embedder: ProviderSpec
        | BaseEmbeddingsProvider[Any]
        | type[BaseEmbeddingsProvider[Any]]
        | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._client: BaseClient | None = None

        warnings.filterwarnings(
            "ignore",
            message=r".*'model_fields'.*is deprecated.*",
            module=r"^chromadb(\.|$)",
        )

        if embedder:
            embedding_function = build_embedder(embedder)  # type: ignore[arg-type]
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

    def save(self, documents: list[Any]) -> None:
        try:
            client = self._get_client()
            collection_name = (
                f"knowledge_{self.collection_name}"
                if self.collection_name
                else "knowledge"
            )
            client.get_or_create_collection(collection_name=collection_name)

            # Accept both old (list[str]) and new (list[dict]) chunk formats
            rag_documents: list[BaseRecord] = _coerce_to_records(documents)
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
