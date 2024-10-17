import contextlib
import io
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional
from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities.paths import db_storage_path
from chromadb.api import ClientAPI


@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ), contextlib.suppress(UserWarning):
        yield
    logger.setLevel(original_level)


class RAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    app: ClientAPI | None = None

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        super().__init__(type, allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents

        self.type = type
        self.embedder_config = embedder_config or self._create_embedding_function()
        self.allow_reset = allow_reset
        self._initialize_app()

    def _initialize_app(self):
        import chromadb

        chroma_client = chromadb.PersistentClient(
            path=f"{db_storage_path()}/{self.type}/{self.agents}"
        )
        self.app = chroma_client

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
        try:
            shutil.rmtree(f"{db_storage_path()}/{self.type}")
            if self.app:
                self.app.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the {self.type} memory: {e}"
            )

    def _create_embedding_function(self):
        import chromadb.utils.embedding_functions as embedding_functions

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
