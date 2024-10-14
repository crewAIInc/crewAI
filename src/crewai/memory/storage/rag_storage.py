import contextlib
import io
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional
from crewai.memory.storage.base_rag_storage import BaseRAGStorage

from crewai.utilities.paths import db_storage_path


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

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        super().__init__(type, allow_reset, embedder_config, crew)
        # TODO: figuiring out OPENAI_API_KEY DEFAULTS

        # if (
        #     not os.getenv("OPENAI_API_KEY")
        #     and not os.getenv("OPENAI_BASE_URL") == "https://api.openai.com/v1"
        # ):
        #     os.environ["OPENAI_API_KEY"] = "fake"

        # agents = crew.agents if crew else []
        # agents = [self._sanitize_role(agent.role) for agent in agents]
        # agents = "_".join(agents)
        # self.agents = agents

        # config = {
        #     "app": {
        #         "config": {"name": type, "collect_metrics": False, "log_level": "ERROR"}
        #     },
        #     "chunker": {
        #         "chunk_size": 5000,
        #         "chunk_overlap": 100,
        #         "length_function": "len",
        #         "min_chunk_size": 150,
        #     },
        #     "vectordb": {
        #         "provider": "chroma",
        #         "config": {
        #             "collection_name": type,
        #             "dir": f"{db_storage_path()}/{type}/{agents}",
        #             "allow_reset": allow_reset,
        #         },
        #     },
        # }

        # if embedder_config:
        #     config["embedder"] = embedder_config
        # self.type = type
        # self.config = config
        self.allow_reset = allow_reset
        self.app: Any = None  # Initialize app attribute
        self._initialize_app()  # Call _initialize_app in the constructor

    def _initialize_app(self):
        import chromadb
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        from chromadb.errors import InvalidCollectionException

        chroma_client = chromadb.PersistentClient(
            path=f"{db_storage_path()}/{self.type}/{self.agents}"
        )
        self.app = chroma_client

        openai_ef = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

        try:
            self.collection = self.app.get_collection(name=self.type)
        except InvalidCollectionException:
            self.collection = self.app.create_collection(
                name=self.type, embedding_function=openai_ef
            )

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()
        self._generate_embedding(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()

        with suppress_logging():
            response = self.collection.query(query_texts=query, n_results=limit)

        results = []
        for i in range(len(response["ids"][0])):
            result = {
                "id": response["ids"][0][i],
                "metadata": response["metadatas"][0][i],
                "document": response["documents"][0][i],
                "score": response["distances"][0][i],
            }
            if result["score"] >= score_threshold:
                results.append(result)

        return results

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> Any:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())],  # Generate a unique ID for each entry
        )

    def reset(self) -> None:
        try:
            shutil.rmtree(f"{db_storage_path()}/{self.type}")
            if hasattr(self, "app"):
                self.app.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the {self.type} memory: {e}"
            )
