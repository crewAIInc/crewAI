import os

from pydantic import BaseModel, ConfigDict, Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.rag.types import SearchResult


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


class Knowledge(BaseModel):
    """
    Knowledge is een verzameling van bronnen en setup voor de vector store om relevante context op te slaan en te bevragen.
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
        **data: object,
    ) -> None:
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
        Bevraag alle kennisbronnen om de meest relevante informatie te vinden.
        Retourneert de top_k meest relevante chunks.

        Gooit:
            ValueError: Als opslag niet is geïnitialiseerd.
        """
        if self.storage is None:
            raise ValueError("Opslag is niet geïnitialiseerd.")

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
        except Exception as e:
            raise e

    def reset(self) -> None:
        if self.storage:
            self.storage.reset()
        else:
            raise ValueError("Opslag is niet geïnitialiseerd.")

    async def aquery(
        self, query: list[str], results_limit: int = 5, score_threshold: float = 0.6
    ) -> list[SearchResult]:
        """Bevraag alle kennisbronnen asynchroon.

        Args:
            query: Lijst van query strings.
            results_limit: Maximaal aantal resultaten om te retourneren.
            score_threshold: Minimale gelijkenisscore voor resultaten.

        Retourneert:
            De top resultaten die overeenkomen met de query.

        Gooit:
            ValueError: Als opslag niet is geïnitialiseerd.
        """
        if self.storage is None:
            raise ValueError("Opslag is niet geïnitialiseerd.")

        return await self.storage.asearch(
            query,
            limit=results_limit,
            score_threshold=score_threshold,
        )

    async def aadd_sources(self) -> None:
        """Voeg alle kennisbronnen asynchroon toe aan opslag."""
        try:
            for source in self.sources:
                source.storage = self.storage
                await source.aadd()
        except Exception as e:
            raise e

    async def areset(self) -> None:
        """Reset de kennisbasis asynchroon."""
        if self.storage:
            await self.storage.areset()
        else:
            raise ValueError("Opslag is niet geïnitialiseerd.")
