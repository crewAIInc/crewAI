"""``RAG.add`` must not hand ``None`` metadata values to ChromaDB.

ChromaDB rejects them with ``TypeError: Cannot convert Python object to
MetadataValue``; the write is wrapped in ``except Exception`` and only logged,
so the documents used to disappear silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crewai_tools.rag.core import RAG


class _RecordingEmbeddingService:
    """Stands in for the provider-backed service: fixed-size vectors, no network."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2]


def _rag(tmp_path: Path) -> RAG:
    rag = RAG(
        collection_name="none_metadata_values",
        persist_directory=str(tmp_path / "chroma"),
    )
    rag._embedding_service = _RecordingEmbeddingService()  # type: ignore[assignment]
    return rag


def test_add_drops_none_metadata_values_and_writes_the_document(
    tmp_path: Path,
) -> None:
    """The document lands, and the ``None`` keys are gone from what ChromaDB stores."""
    rag = _rag(tmp_path)

    rag.add(
        content="A transcript of a video",
        metadata={"title": "Video", "description": None, "author": None},
    )

    result = rag._collection.get(include=["metadatas", "documents"])
    assert len(result["ids"]) == 1
    metadata: dict[str, Any] = result["metadatas"][0]
    assert metadata["title"] == "Video"
    assert "description" not in metadata
    assert "author" not in metadata


def test_add_keeps_supported_values_unchanged(tmp_path: Path) -> None:
    """Values ChromaDB already accepts are passed through, including falsy ones."""
    rag = _rag(tmp_path)

    rag.add(
        content="Another transcript",
        metadata={"title": "Clean", "views": 0, "live": False},
    )

    result = rag._collection.get(include=["metadatas"])
    metadata = result["metadatas"][0]
    assert metadata["title"] == "Clean"
    assert metadata["views"] == 0
    assert metadata["live"] is False
