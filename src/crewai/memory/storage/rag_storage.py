import contextlib
import io
import logging
from typing import Any, Dict

from embedchain import App
from embedchain.llm.base import BaseLlm

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


class FakeLLM(BaseLlm):
    pass


class RAGStorage(Storage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    def __init__(self, type, allow_reset=True, embedder_config=None):
        super().__init__()
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
                    "dir": f"{db_storage_path()}/{type}",
                    "allow_reset": allow_reset,
                },
            },
        }

        if embedder_config:
            config["embedder"] = embedder_config

        self.app = App.from_config(config=config)
        self.app.llm = FakeLLM()
        if allow_reset:
            self.app.reset()

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        self._generate_embedding(value, metadata)

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: dict = None,
        score_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        with suppress_logging():
            results = (
                self.app.search(query, limit, where=filter)
                if filter
                else self.app.search(query, limit)
            )
        return [r for r in results if r["metadata"]["score"] >= score_threshold]

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> Any:
        with suppress_logging():
            self.app.add(text, data_type="text", metadata=metadata)
