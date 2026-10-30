"""Chunk identity must preserve every occurrence in the exported RAG adapter."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from crewai_tools.rag import core
from crewai_tools.rag.chunkers.base_chunker import BaseChunker
from crewai_tools.rag.core import RAG
from crewai_tools.rag.data_types import DataType


class _OfflineEmbeddings:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _LineChunker(BaseChunker):
    def chunk(self, text: str) -> list[str]:
        return text.splitlines()


@pytest.fixture
def rag(monkeypatch: pytest.MonkeyPatch) -> Iterator[RAG]:
    monkeypatch.setattr(core, "EmbeddingService", _OfflineEmbeddings)
    instance = RAG(collection_name=f"rag_chunk_ids_{uuid4().hex}")
    yield instance
    instance.delete_collection()


def _chunks_for_source(rag: RAG, source: Path) -> list[str]:
    records = rag._collection.get(
        where={"source": str(source)}, include=["documents"]
    )
    return records["documents"] or []


def test_shared_chunk_is_stored_for_each_source(rag: RAG, tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("Shared boilerplate\nUnique first", encoding="utf-8")
    second.write_text("Shared boilerplate\nUnique second", encoding="utf-8")

    rag.add(first, data_type=DataType.TEXT_FILE, chunker=_LineChunker())
    rag.add(second, data_type=DataType.TEXT_FILE, chunker=_LineChunker())

    assert Counter(_chunks_for_source(rag, first)) == Counter(
        ["Shared boilerplate", "Unique first"]
    )
    assert Counter(_chunks_for_source(rag, second)) == Counter(
        ["Shared boilerplate", "Unique second"]
    )


def test_shared_chunk_survives_loader_doc_id_collision(rag: RAG, tmp_path: Path) -> None:
    # The loader hashes source path + content without a separator: a + bc == ab + c.
    first = tmp_path / "a"
    second = tmp_path / "ab"
    first.write_text("bc\nShared", encoding="utf-8")
    second.write_text("c\nShared", encoding="utf-8")

    rag.add(first, data_type=DataType.TEXT_FILE, chunker=_LineChunker())
    rag.add(second, data_type=DataType.TEXT_FILE, chunker=_LineChunker())

    assert Counter(_chunks_for_source(rag, first)) == Counter(["bc", "Shared"])
    assert Counter(_chunks_for_source(rag, second)) == Counter(["c", "Shared"])


def test_repeated_chunks_survive_reindexing(rag: RAG, tmp_path: Path) -> None:
    source = tmp_path / "document.txt"
    source.write_text("Old content", encoding="utf-8")
    rag.add(source, data_type=DataType.TEXT_FILE, chunker=_LineChunker())

    source.write_text("Repeated paragraph\nRepeated paragraph", encoding="utf-8")
    rag.add(source, data_type=DataType.TEXT_FILE, chunker=_LineChunker())

    assert Counter(_chunks_for_source(rag, source)) == Counter(
        ["Repeated paragraph", "Repeated paragraph"]
    )

    rag.add(source, data_type=DataType.TEXT_FILE, chunker=_LineChunker())
    assert rag._collection.count() == 2
