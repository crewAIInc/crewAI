import contextlib
import io
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from crewai.memory.storage.interface import Storage
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


class RAGStorage(Storage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        super().__init__()
        if (
            not os.getenv("OPENAI_API_KEY")
            and not os.getenv("OPENAI_BASE_URL") == "https://api.openai.com/v1"
        ):
            os.environ["OPENAI_API_KEY"] = "fake"

        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)

        config = {
            "app": {
                "config": {"name": type, "collect_metrics": False, "log_level": "ERROR"}
            },
            "chunker": {
                "chunk_size": 5000,
                "chunk_overlap": 100,
                "length_function": "len",
                "min_chunk_size": 150,
            },
            "vectordb": {
                "provider": "chroma",
                "config": {
                    "collection_name": type,
                    "dir": f"{db_storage_path()}/{type}/{agents}",
                    "allow_reset": allow_reset,
                },
            },
        }

        if embedder_config:
            config["embedder"] = embedder_config
        self.type = type
        self.config = config
        self.allow_reset = allow_reset

    def _initialize_app(self):
        from embedchain import App
        from embedchain.llm.base import BaseLlm

        class FakeLLM(BaseLlm):
            pass

        self.app = App.from_config(config=self.config)
        self.app.llm = FakeLLM()
        if self.allow_reset:
            self.app.reset()

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app"):
            self._initialize_app()
        self._generate_embedding(value, metadata)

    def search(  # type: ignore # BUG?: Signature of "search" incompatible with supertype "Storage"
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()
        from embedchain.vectordb.chroma import InvalidDimensionException

        with suppress_logging():
            try:
                results = (
                    self.app.search(query, limit, where=filter)
                    if filter
                    else self.app.search(query, limit)
                )
            except InvalidDimensionException:
                self.app.reset()
                return []
        return [r for r in results if r["metadata"]["score"] >= score_threshold]

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> Any:
        if not hasattr(self, "app"):
            self._initialize_app()
        from embedchain.models.data_type import DataType

        self.app.add(text, data_type=DataType.TEXT, metadata=metadata)

    def reset(self) -> None:
        try:
            shutil.rmtree(f"{db_storage_path()}/{self.type}")
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the {self.type} memory: {e}"
            )
