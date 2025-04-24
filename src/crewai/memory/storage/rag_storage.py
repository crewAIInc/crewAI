import contextlib
import io
import logging
import os
import shutil
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Type checking imports that don't cause runtime imports
if TYPE_CHECKING:
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.config import Settings

from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.chromadb import sanitize_collection_name
from crewai.utilities.logger import Logger
from crewai.utilities.paths import db_storage_path


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


class RAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    collection: Optional[Any] = None
    collection_name: Optional[str] = "memory"
    app: Optional[Any] = None

    def __init__(
        self,
        type: str = "memory",
        allow_reset: bool = True,
        embedder_config: Optional[Dict[str, Any]] = None,
        crew: Any = None,
        collection_name: Optional[str] = None,
    ):
        super().__init__(type, allow_reset, embedder_config, crew)
        self.collection_name = collection_name or type
        self._set_embedder_config(embedder_config)

    def save(
        self,
        value: Any,
        metadata: Dict[str, Any],
    ) -> None:
        with suppress_logging():
            if not self.collection:
                self._initialize_app()

            if isinstance(value, list):
                documents = value
                metadatas = [metadata] * len(value) if metadata else None
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            else:
                documents = [value]
                metadatas = [metadata] if metadata else None
                ids = [str(uuid.uuid4())]

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        with suppress_logging():
            if not hasattr(self, "collection") or not self.collection:
                self._initialize_app()

            if isinstance(query, str):
                query = [query]

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
                if result["score"] >= score_threshold:
                    results.append(result)
            return results

    def _initialize_app(self):
        # Import chromadb here to avoid importing at module level
        import chromadb
        from chromadb.config import Settings
        
        base_path = os.path.join(db_storage_path(), "memory")
        chroma_client = chromadb.PersistentClient(
            path=base_path,
            settings=Settings(allow_reset=self.allow_reset),
        )

        self.app = chroma_client

        try:
            collection_name = (
                f"memory_{self.collection_name}"
                if self.collection_name
                else "memory"
            )
            if self.app:
                self.collection = self.app.get_or_create_collection(
                    name=sanitize_collection_name(collection_name),
                    embedding_function=self.embedder,
                )
            else:
                raise Exception("Vector Database Client not initialized")
        except Exception:
            raise Exception("Failed to create or get collection")

    def initialize_rag_storage(self):
        self._initialize_app()

    def reset(self) -> None:
        # Import chromadb here to avoid importing at module level
        import chromadb
        from chromadb.config import Settings
        
        base_path = os.path.join(db_storage_path(), "memory")
        if not self.app:
            self.app = chromadb.PersistentClient(
                path=base_path,
                settings=Settings(allow_reset=True),
            )

        self.app.reset()
        shutil.rmtree(base_path)
        self.app = None
        self.collection = None

    def _generate_embedding(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        if not hasattr(self, "collection") or not self.collection:
            self._initialize_app()

        id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[id],
        )
        return id

    def _sanitize_role(self, role: str) -> str:
        """Sanitize role name for use in file names."""
        return role.lower().replace(" ", "_").replace("\n", "").replace("/", "_")

    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    def _set_embedder_config(self, embedder_config: Optional[Dict[str, Any]] = None) -> None:
        """Set the embedding configuration for the RAG storage.

        Args:
            embedder_config (Optional[Dict[str, Any]]): Configuration dictionary for the embedder.
                If None or empty, defaults to the default embedding function.
        """
        self.embedder = (
            EmbeddingConfigurator().configure_embedder(embedder_config)
            if embedder_config
            else self._create_default_embedding_function()
        )

    def _build_storage_file_name(self, role_name: str) -> str:
        """Build storage file name from role name."""
        return f"{self._sanitize_role(role_name)}_memory"
