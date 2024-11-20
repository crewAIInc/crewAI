import contextlib
import io
import logging
import chromadb
import os
from crewai.utilities.paths import db_storage_path
from typing import Optional, List
from typing import Dict, Any
from crewai.utilities import EmbeddingConfigurator
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
import hashlib


@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.suppress(UserWarning),
    ):
        yield
    logger.setLevel(original_level)


class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    collection: Optional[chromadb.Collection] = None

    def __init__(self, embedder_config: Optional[Dict[str, Any]] = None):
        self._initialize_app(embedder_config or {})

    def search(
        self,
        query: List[str],
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        with suppress_logging():
            if self.collection:
                fetched = self.collection.query(
                    query_texts=query,
                    n_results=limit,
                    where=filter,
                )
                results = []
                for i in range(len(fetched["ids"][0])):  # type: ignore
                    result = {
                        "id": fetched["ids"][0][i],  # type: ignore
                        "metadata": fetched["metadatas"][0][i],  # type: ignore
                        "context": fetched["documents"][0][i],  # type: ignore
                        "score": fetched["distances"][0][i],  # type: ignore
                    }
                    if result["score"] >= score_threshold:  # type: ignore
                        results.append(result)
                return results
            else:
                raise Exception("Collection not initialized")

    def _initialize_app(self, embedder_config: Optional[Dict[str, Any]] = None):
        import chromadb
        from chromadb.config import Settings

        self._set_embedder_config(embedder_config)

        chroma_client = chromadb.PersistentClient(
            path=f"{db_storage_path()}/knowledge",
            settings=Settings(allow_reset=True),
        )

        self.app = chroma_client

        try:
            self.collection = self.app.get_or_create_collection(name="knowledge")
        except Exception:
            raise Exception("Failed to create or get collection")

    def reset(self):
        if self.app:
            self.app.reset()

    def save(
        self, documents: List[str], metadata: Dict[str, Any] | List[Dict[str, Any]]
    ):
        if self.collection:
            metadatas = [metadata] if isinstance(metadata, dict) else metadata

            ids = [
                hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in documents
            ]

            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            raise Exception("Collection not initialized")

    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    def _set_embedder_config(
        self, embedder_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the embedding configuration for the knowledge storage.

        Args:
            embedder_config (Optional[Dict[str, Any]]): Configuration dictionary for the embedder.
                If None or empty, defaults to the default embedding function.
        """
        self.embedder_config = (
            EmbeddingConfigurator().configure_embedder(embedder_config)
            if embedder_config
            else self._create_default_embedding_function()
        )
