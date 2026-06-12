"""Embedding dimension mismatch must fail loudly with migration guidance.

The default embedder changed from text-embedding-3-small (1536 dims) to
text-embedding-3-large (3072 dims); stores created before the upgrade must
not silently zero-fill vectors or return empty search results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord


@pytest.fixture
def lancedb_path(tmp_path: Path) -> Path:
    return tmp_path / "mem"


def _record(dim: int, content: str = "test") -> MemoryRecord:
    return MemoryRecord(content=content, scope="/foo", embedding=[0.1] * dim)


def test_lancedb_save_mismatch_raises(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    storage.save([_record(4)])

    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
        storage.save([_record(8, "new embedder output")])

    message = str(exc_info.value)
    assert "4-dimensional" in message
    assert "8-dimensional" in message
    assert "crewai reset-memories --memory" in message
    assert "text-embedding-3-small" in message


def test_lancedb_mixed_batch_mismatch_raises(lancedb_path: Path) -> None:
    """A single save() batch with inconsistent dimensions must be rejected."""
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    storage.save([_record(4)])

    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.save([_record(4), _record(8, "stray dimension")])


def test_lancedb_mixed_batch_on_fresh_store_raises(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path))
    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.save([_record(4), _record(8)])


def test_lancedb_search_mismatch_raises(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    storage.save([_record(4)])

    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.search([0.1] * 8)


def test_lancedb_update_mismatch_raises(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    record = _record(4)
    storage.save([record])

    stale = MemoryRecord(
        id=record.id, content="updated", scope="/foo", embedding=[0.1] * 8
    )
    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.update(stale)


def test_lancedb_reopened_store_detects_mismatch(lancedb_path: Path) -> None:
    """The upgrade scenario: an old store reopened with a new embedder."""
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    old = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    old.save([_record(4)])

    reopened = LanceDBStorage(path=str(lancedb_path))
    with pytest.raises(EmbeddingDimensionMismatchError):
        reopened.save([_record(8)])
    with pytest.raises(EmbeddingDimensionMismatchError):
        reopened.search([0.1] * 8)


def test_lancedb_matching_dim_still_works(lancedb_path: Path) -> None:
    from crewai.memory.storage.lancedb_storage import LanceDBStorage

    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    storage.save([_record(4)])
    storage.save([_record(4, "second")])

    assert len(storage.search([0.1] * 4, limit=5)) == 2
