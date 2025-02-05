import contextlib
import io
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from chromadb.api import ClientAPI, Collection
from chromadb.api.types import Documents, Embeddings, Metadatas

from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.constants import MAX_FILE_NAME_LENGTH
from crewai.utilities.paths import db_storage_path
from crewai.utilities.exceptions.embedding_exceptions import (
    EmbeddingConfigurationError,
    EmbeddingInitializationError
)


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
    """RAG-based Storage implementation using ChromaDB for vector storage and retrieval.

    This class extends BaseRAGStorage to handle embeddings for memory entries,
    improving search efficiency through vector similarity.

    Attributes:
        app: ChromaDB client instance
        collection: ChromaDB collection for storing embeddings
        type: Type of memory storage
        allow_reset: Whether memory reset is allowed
        path: Custom storage path for the database
    """

    app: ClientAPI | None = None
    collection: Any = None

    def __init__(
        self, type: str, allow_reset: bool = True, embedder_config: Dict[str, Any] | None = None, crew: Any = None, path: str | None = None
    ):
        super().__init__(type, allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents
        self.storage_file_name = self._build_storage_file_name(type, agents)

        self.type = type
        self.allow_reset = allow_reset
        self.path = path
        self._initialize_app()

    def _set_embedder_config(self):
        configurator = EmbeddingConfigurator()
        self.embedder_config = configurator.configure_embedder(self.embedder_config)

    def _initialize_app(self) -> None:
        """Initialize the ChromaDB client and collection.

        Raises:
            RuntimeError: If ChromaDB client initialization fails
            EmbeddingConfigurationError: If embedding configuration is invalid
            EmbeddingInitializationError: If embedding function fails to initialize
        """
        import chromadb
        from chromadb.config import Settings

        self._set_embedder_config()
        try:
            chroma_client = chromadb.PersistentClient(
                path=self.path if self.path else self.storage_file_name,
                settings=Settings(allow_reset=self.allow_reset),
            )
            self.app = chroma_client
            if not self.app:
                raise RuntimeError("Failed to initialize ChromaDB client")

            try:
                self.collection = self.app.get_collection(
                    name=self.type, embedding_function=self.embedder_config
                )
            except Exception:
                self.collection = self.app.create_collection(
                    name=self.type, embedding_function=self.embedder_config
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {str(e)}")

        self.app = chroma_client
        if not self.app:
            raise RuntimeError("Failed to initialize ChromaDB client")

        try:
            self.collection = self.app.get_collection(
                name=self.type, embedding_function=self.embedder_config
            )
        except Exception:
            self.collection = self.app.create_collection(
                name=self.type, embedding_function=self.embedder_config
            )

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def _build_storage_file_name(self, type: str, file_name: str) -> str:
        """
        Ensures file name does not exceed max allowed by OS
        """
        base_path = f"{db_storage_path()}/{type}"

        if len(file_name) > MAX_FILE_NAME_LENGTH:
            logging.warning(
                f"Trimming file name from {len(file_name)} to {MAX_FILE_NAME_LENGTH} characters."
            )
            file_name = file_name[:MAX_FILE_NAME_LENGTH]

        return f"{base_path}/{file_name}"

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()
        try:
            self._generate_embedding(value, metadata)
        except Exception as e:
            logging.error(f"Error during {self.type} save: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()

        try:
            with suppress_logging():
                response = self.collection.query(query_texts=query, n_results=limit)

            results = []
            for i in range(len(response["ids"][0])):
                result = {
                    "id": response["ids"][0][i],
                    "metadata": response["metadatas"][0][i],
                    "context": response["documents"][0][i],
                    "score": response["distances"][0][i],
                }
                if result["score"] >= score_threshold:
                    results.append(result)

            return results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> None:  # type: ignore
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())],
        )

    def reset(self) -> None:
        """Reset the memory storage by clearing the database and removing files.

        Raises:
            RuntimeError: If memory reset fails and allow_reset is False
            EmbeddingConfigurationError: If embedding configuration is invalid during reinitialization
        """
        try:
            if self.app:
                self.app.reset()
                storage_path = self.path if self.path else db_storage_path()
                db_dir = os.path.join(storage_path, self.type)
                if os.path.exists(db_dir):
                    shutil.rmtree(db_dir)
                self.app = None
                self.collection = None
        except Exception as e:
            if "attempt to write a readonly database" in str(e):
                # Ignore this specific error as it's expected in some environments
                pass
            else:
                if not self.allow_reset:
                    raise RuntimeError(f"Failed to reset {self.type} memory: {str(e)}")
                logging.error(f"Error during {self.type} memory reset: {str(e)}")
