import contextlib
import io
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional

from chromadb.api import ClientAPI

from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.constants import MAX_FILE_NAME_LENGTH
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

    app: ClientAPI | None = None

    def __init__(
        self, type, allow_reset=True, embedder_config=None, crew=None, path=None
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

    def _initialize_app(self):
        import chromadb
        from chromadb.config import Settings

        self._set_embedder_config()
        chroma_client = chromadb.PersistentClient(
            path=self.path if self.path else self.storage_file_name,
            settings=Settings(allow_reset=self.allow_reset),
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
        recency_weight: float = 0.3,
        time_decay_days: float = 1.0,
    ) -> List[Any]:
        """
        Search for entries in the storage based on semantic similarity and recency.
        
        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            filter: Optional filter to apply to the search.
            score_threshold: Minimum score threshold for results.
            recency_weight: Weight given to recency vs. semantic similarity (0.0-1.0).
                Higher values prioritize recent memories more strongly.
            time_decay_days: Number of days over which recency factor decays to zero.
                Smaller values make older memories lose relevance faster.
        
        Returns:
            List of search results, each containing id, metadata, context, and score.
            Results are sorted by combined semantic similarity and recency score.
        """
        if not hasattr(self, "app"):
            self._initialize_app()

        try:
            with suppress_logging():
                response = self.collection.query(query_texts=query, n_results=limit * 2)  # Get more results to allow for recency filtering

            results = []
            for i in range(len(response["ids"][0])):
                result = {
                    "id": response["ids"][0][i],
                    "metadata": response["metadatas"][0][i],
                    "context": response["documents"][0][i],
                    "score": response["distances"][0][i],
                }
                
                # Apply recency boost if timestamp exists in metadata
                if "timestamp" in result["metadata"]:
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(result["metadata"]["timestamp"])
                        now = datetime.now()
                        # Calculate recency factor (newer = higher score)
                        time_diff_seconds = (now - timestamp).total_seconds()
                        recency_factor = max(0, 1 - (time_diff_seconds / (time_decay_days * 24 * 60 * 60)))
                        # Adjust score with recency factor
                        result["score"] = result["score"] * (1 - recency_weight) + recency_factor * recency_weight
                    except (ValueError, TypeError):
                        pass  # If timestamp parsing fails, use original score

                if result["score"] >= score_threshold:
                    results.append(result)

            # Sort by adjusted score (higher is better)
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]  # Return only the requested number of results
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
            if self.app:
                self.app.reset()
                shutil.rmtree(f"{db_storage_path()}/{self.type}")
                self.app = None
                self.collection = None
        except Exception as e:
            if "attempt to write a readonly database" in str(e):
                # Ignore this specific error
                pass
            else:
                raise Exception(
                    f"An error occurred while resetting the {self.type} memory: {e}"
                )

    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
