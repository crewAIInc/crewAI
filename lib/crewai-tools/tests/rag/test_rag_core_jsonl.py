"""JSONL ingestion through the real loader and chunker, with no external services."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from crewai_tools.rag.core import RAG
from crewai_tools.rag.data_types import DataType
import pytest


class FakeEmbeddings:
    """Return deterministic vectors and capture the embedded text."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeCollection:
    """Capture stored documents without starting ChromaDB."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
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
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)


@pytest.fixture
def rag() -> tuple[RAG, FakeCollection, FakeEmbeddings]:
    """Use real RAG initialization with fake storage and embeddings."""
    collection = FakeCollection()
    embeddings = FakeEmbeddings()
    client = Mock()
    client.get_or_create_collection.return_value = collection
    with (
        patch("crewai_tools.rag.core.chromadb.Client", return_value=client),
        patch("crewai_tools.rag.core.EmbeddingService", return_value=embeddings),
    ):
        instance = RAG()
    return instance, collection, embeddings


@pytest.mark.parametrize(
    ("extension", "data_type"),
    [
        (".jsonl", None),
        (".ndjson", None),
        (".txt", "jsonl"),
        (".txt", DataType.JSONL),
    ],
)
def test_add_jsonl_stores_records(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings],
    tmp_path: Path,
    extension: str,
    data_type: str | DataType | None,
) -> None:
    """Explicit selection and discovery preserve JSON values and record order."""
    instance, collection, embeddings = rag
    lines = ['{"name": "café", "active": true}', '{"items": [1, null]}', '"last"']
    path = tmp_path / f"records{extension}"
    path.write_text("\n".join(lines), encoding="utf-8")

    instance.add(path, data_type=data_type, metadata={"dataset": "example"})

    stored_lines = [
        line
        for document in collection.documents
        for line in document.splitlines()
        if line.strip()
    ]
    assert stored_lines == lines
    assert [json.loads(line) for line in stored_lines] == [
        {"name": "café", "active": True},
        {"items": [1, None]},
        "last",
    ]
    assert embeddings.texts == collection.documents
    assert collection.embeddings == [[1.0, 0.0]] * len(collection.documents)
    assert len(collection.ids) == len(collection.documents) > 0
    for index, metadata in enumerate(collection.metadatas):
        assert metadata["data_type"] == "jsonl"
        assert metadata["source"] == str(path)
        assert metadata["dataset"] == "example"
        assert metadata["chunk_index"] == index
        assert metadata["doc_id"]


def test_add_inline_jsonl(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings],
) -> None:
    """Inline JSONL requires explicit selection, not a whole-document JSON parser."""
    instance, collection, _ = rag
    instance.add('{"a": 1}\n{"b": 2}', data_type="jsonl")
    assert [
        json.loads(line)
        for document in collection.documents
        for line in document.splitlines()
        if line.strip()
    ] == [{"a": 1}, {"b": 2}]


def test_invalid_jsonl_is_not_stored(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings], tmp_path: Path
) -> None:
    """Malformed records fail before any embedding or storage."""
    instance, collection, embeddings = rag
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"a": 1}\nnot json', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        instance.add(path)
    assert collection.documents == []
    assert embeddings.texts == []


def test_empty_jsonl_is_not_stored(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings], tmp_path: Path
) -> None:
    """Empty loader output does not produce embeddings or documents."""
    instance, collection, embeddings = rag
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    instance.add(path)
    assert collection.documents == []
    assert embeddings.texts == []
