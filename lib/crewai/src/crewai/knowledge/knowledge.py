import os
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import (
    TextFileKnowledgeSource,
)
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.rag.types import SearchResult


_KNOWN_SOURCES: dict[str, type[BaseKnowledgeSource]] = {
    "string": StringKnowledgeSource,
    "docling": CrewDoclingSource,
    "csv": CSVKnowledgeSource,
    "excel": ExcelKnowledgeSource,
    "json": JSONKnowledgeSource,
    "pdf": PDFKnowledgeSource,
    "text_file": TextFileKnowledgeSource,
}


def _resolve_knowledge_sources(value: Any) -> Any:
    """Coerce list of dicts into typed BaseKnowledgeSource subclasses via source_type.

    Pass-through for anything else (existing instances, mocks).
    """
    if not isinstance(value, list):
        return value
    resolved: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            tag = item.get("source_type")
            cls = _KNOWN_SOURCES.get(tag) if isinstance(tag, str) else None
            if cls is None:
                resolved.append(item)
            else:
                resolved.append(cls.model_validate(item))
        else:
            resolved.append(item)
    return resolved


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # removes logging from fastembed


def _serialize_embedder_spec(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseEmbeddingsProvider):
        return value.model_dump(mode="json")
    if isinstance(value, type) and issubclass(value, BaseEmbeddingsProvider):
        return {"provider_class": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, dict):
        return value
    raise TypeError(
        f"Cannot serialize embedder of type {type(value).__name__}: "
        "expected ProviderSpec dict, BaseEmbeddingsProvider instance, or subclass."
    )


class Knowledge(BaseModel):
    """
    Knowledge is a collection of sources and setup for the vector store to save and query relevant context.
    Args:
        sources: list[BaseKnowledgeSource] = Field(default_factory=list)
        storage: KnowledgeStorage | None = Field(default=None)
        embedder: EmbedderConfig | None = None
    """

    sources: Annotated[
        list[BaseKnowledgeSource],
        BeforeValidator(_resolve_knowledge_sources),
    ] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: KnowledgeStorage | None = Field(default=None)
    embedder: Annotated[
        EmbedderConfig | None,
        PlainSerializer(
            _serialize_embedder_spec, return_type=dict | None, when_used="json"
        ),
    ] = None
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
            raise ValueError("Storage is not initialized.")

    async def aquery(
        self, query: list[str], results_limit: int = 5, score_threshold: float = 0.6
    ) -> list[SearchResult]:
        """Query across all knowledge sources asynchronously.

        Args:
            query: List of query strings.
            results_limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            The top results matching the query.

        Raises:
            ValueError: If storage is not initialized.
        """
        if self.storage is None:
            raise ValueError("Storage is not initialized.")

        return await self.storage.asearch(
            query,
            limit=results_limit,
            score_threshold=score_threshold,
        )

    async def aadd_sources(self) -> None:
        """Add all knowledge sources to storage asynchronously."""
        try:
            for source in self.sources:
                source.storage = self.storage
                await source.aadd()
        except Exception as e:
            raise e

    async def areset(self) -> None:
        """Reset the knowledge base asynchronously."""
        if self.storage:
            await self.storage.areset()
        else:
            raise ValueError("Storage is not initialized.")
