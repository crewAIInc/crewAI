import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


class Knowledge(BaseModel):
    """Knowledge is a collection of sources and setup for the vector store to save and query relevant context.

    Args:
        sources: List[BaseKnowledgeSource] = Field(default_factory=list)
        storage: Optional[KnowledgeStorage] = Field(default=None)
        embedder: Optional[Dict[str, Any]] = None.

    """

    sources: list[BaseKnowledgeSource] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: KnowledgeStorage | None = Field(default=None)
    embedder: dict[str, Any] | None = None
    collection_name: str | None = None

    def __init__(
        self,
        collection_name: str,
        sources: list[BaseKnowledgeSource],
        embedder: dict[str, Any] | None = None,
        storage: KnowledgeStorage | None = None,
        **data,
    ) -> None:
        super().__init__(**data)
        if storage:
            self.storage = storage
        else:
            self.storage = KnowledgeStorage(
                embedder=embedder, collection_name=collection_name,
            )
        self.sources = sources
        self.storage.initialize_knowledge_storage()

    def query(
        self, query: list[str], results_limit: int = 3, score_threshold: float = 0.35,
    ) -> list[dict[str, Any]]:
        """Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.

        Raises:
            ValueError: If storage is not initialized.

        """
        if self.storage is None:
            msg = "Storage is not initialized."
            raise ValueError(msg)

        return self.storage.search(
            query,
            limit=results_limit,
            score_threshold=score_threshold,
        )

    def add_sources(self) -> None:
        try:
            for source in self.sources:
                source.storage = self.storage
                source.add()
        except Exception:
            raise

    def reset(self) -> None:
        if self.storage:
            self.storage.reset()
        else:
            msg = "Storage is not initialized."
            raise ValueError(msg)
