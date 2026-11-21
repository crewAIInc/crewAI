"""Loader metadata reaches stored chunks through RAG.add, with no external services."""

from typing import Any
from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.chunkers.base_chunker import BaseChunker
from crewai_tools.rag.core import RAG
from crewai_tools.rag.source_content import SourceContent
import pytest


class FakeLoader(BaseLoader):
    """Return a fixed result so its metadata can be inspected after ingestion."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        super().__init__()
        self.result = LoaderResult(
            content="first\n\nsecond",
            source="loader-source",
            metadata=metadata,
            doc_id="loader-doc-id",
        )

    def load(self, content: SourceContent, **kwargs: Any) -> LoaderResult:
        return self.result


class FakeChunker(BaseChunker):
    """Split on blank lines so each test sees a known number of chunks."""

    def chunk(self, text: str) -> list[str]:
        return text.split("\n\n")


class FakeEmbeddings:
    """Return deterministic vectors instead of calling an embedding provider."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeCollection:
    """Capture stored metadata without starting ChromaDB."""

    def __init__(self) -> None:
        self.metadatas: list[dict[str, Any]] = []

    def get(self, *, where: dict[str, Any], limit: int) -> dict[str, Any]:
        return {"metadatas": []}

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.metadatas.extend(metadatas)


@pytest.fixture
def rag() -> tuple[RAG, FakeCollection]:
    """Use real RAG initialization with fake storage and embeddings."""
    collection = FakeCollection()
    client = Mock()
    client.get_or_create_collection.return_value = collection
    with (
        patch("crewai_tools.rag.core.chromadb.Client", return_value=client),
        patch("crewai_tools.rag.core.EmbeddingService", return_value=FakeEmbeddings()),
    ):
        instance = RAG()
    return instance, collection


def test_add_stores_loader_metadata_with_caller_metadata(
    rag: tuple[RAG, FakeCollection],
) -> None:
    """Scalar loader fields are kept on every chunk next to caller fields."""
    instance, collection = rag
    loader = FakeLoader({"format": "json", "size": 1, "valid": True})

    instance.add(
        "ignored", metadata={"tag": "user"}, loader=loader, chunker=FakeChunker()
    )

    assert len(collection.metadatas) == 2
    for index, metadata in enumerate(collection.metadatas):
        assert metadata == {
            "format": "json",
            "size": 1,
            "valid": True,
            "tag": "user",
            "chunk_index": index,
            "data_type": "text",
            "source": "loader-source",
            "doc_id": "loader-doc-id",
        }


def test_caller_overrides_loader_but_not_generated_fields(
    rag: tuple[RAG, FakeCollection],
) -> None:
    """Caller wins over descriptive loader fields; generated fields win over both."""
    instance, collection = rag
    generated = {
        "source": "spoofed",
        "doc_id": "spoofed",
        "data_type": "spoofed",
        "chunk_index": 99,
    }
    loader = FakeLoader({"format": "json", **generated})

    instance.add(
        "ignored",
        metadata={"format": "custom", **generated},
        loader=loader,
        chunker=FakeChunker(),
    )

    for index, metadata in enumerate(collection.metadatas):
        assert metadata["format"] == "custom"
        assert metadata["source"] == "loader-source"
        assert metadata["doc_id"] == "loader-doc-id"
        assert metadata["data_type"] == "text"
        assert metadata["chunk_index"] == index


def test_add_does_not_mutate_metadata_inputs(
    rag: tuple[RAG, FakeCollection],
) -> None:
    """Neither the caller's dictionary nor the loader result is modified."""
    instance, _ = rag
    loader_metadata = {"format": "json", "tags": ["a", "b"], "missing": None}
    caller_metadata = {"tag": "user"}
    loader = FakeLoader(loader_metadata)

    instance.add(
        "ignored", metadata=caller_metadata, loader=loader, chunker=FakeChunker()
    )

    assert caller_metadata == {"tag": "user"}
    assert loader.result.metadata == {
        "format": "json",
        "tags": ["a", "b"],
        "missing": None,
    }


def test_unsupported_loader_values_are_sanitized_for_storage(
    rag: tuple[RAG, FakeCollection],
) -> None:
    """Loader values ChromaDB rejects are converted or dropped like elsewhere."""
    instance, collection = rag
    loader = FakeLoader(
        {"tags": ["a", "b"], "info": {"k": 1}, "missing": None, "score": 0.5}
    )

    instance.add("ignored", loader=loader, chunker=FakeChunker())

    metadata = collection.metadatas[0]
    assert metadata["tags"] == "a | b"
    assert metadata["info"] == "{'k': 1}"
    assert metadata["score"] == 0.5
    assert "missing" not in metadata
