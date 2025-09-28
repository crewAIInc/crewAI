import os

from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.rag.types import SearchResult

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


class Knowledge(BaseModel):
    """
    Knowledge is a collection of sources and setup for the vector store to save and query relevant context.
    Args:
        sources: list[BaseKnowledgeSource] = Field(default_factory=list)
        storage: KnowledgeStorage | None = Field(default=None)
        embedder: EmbedderConfig | None = None
    """

    sources: list[BaseKnowledgeSource] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: KnowledgeStorage | None = Field(default=None)
    embedder: EmbedderConfig | None = None
    collection_name: str | None = None

    def __init__(
        self,
        collection_name: str,
        sources: list[BaseKnowledgeSource],
        embedder: EmbedderConfig | None = None,
        storage: KnowledgeStorage | None = None,
        **data,
    ):
        super().__init__(**data)
        if storage:
            self.storage = storage
        else:
            self.storage = KnowledgeStorage(
                embedder=embedder, collection_name=collection_name
            )
        self.sources = sources

    def query(
        self, query: list[str], results_limit: int = 5, score_threshold: float = 0.6
    ) -> list[SearchResult]:
        """
        Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.

        Raises:
            ValueError: If storage is not initialized.
        """
        if self.storage is None:
            raise ValueError("Storage is not initialized.")

        return self.storage.search(
            query,
            limit=results_limit,
            score_threshold=score_threshold,
        )

    def add_sources(self):
        try:
            for source in self.sources:
                source.storage = self.storage
                source.add()
        except Exception as e:
            raise e

    def reset(self) -> None:
        if self.storage:
            self.storage.reset()
        else:
            raise ValueError("Storage is not initialized.")
