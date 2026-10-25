"""Regression tests for record_ids intersection in LanceDBStorage.delete().

record_ids must intersect with the other filters (older_than, categories,
metadata_filter), matching QdrantEdgeStorage's semantics, instead of being
ignored by the filtered paths.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord

DIM = 4


def _record(rid: str, categories: list[str] | None = None, **overrides) -> MemoryRecord:
    record = MemoryRecord(
        id=rid,
        content=f"content {rid}",
        scope="/proj",
        categories=categories or [],
        embedding=[0.1] * DIM,
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


@pytest.fixture()
def storage(tmp_path: Path) -> LanceDBStorage:
    return LanceDBStorage(path=tmp_path, vector_dim=DIM, compact_every=0)


def test_record_ids_with_older_than_keeps_new_records(storage: LanceDBStorage) -> None:
    fresh = _record("fresh")
    old = _record("old")
    old.created_at = datetime.utcnow() - timedelta(days=30)
    storage.save([fresh, old])

    deleted = storage.delete(record_ids=[fresh.id, old.id], older_than=datetime.utcnow() - timedelta(days=1))

    assert deleted == 1
    assert storage.count() == 1
    remaining = storage.search(fresh.embedding, limit=10)
    assert [record.id for record, _ in remaining] == ["fresh"]


def test_record_ids_with_older_than_deletes_old_records(storage: LanceDBStorage) -> None:
    old = _record("old")
    old.created_at = datetime.utcnow() - timedelta(days=30)
    storage.save([old])

    deleted = storage.delete(record_ids=[old.id], older_than=datetime.utcnow() - timedelta(days=1))

    assert deleted == 1
    assert storage.count() == 0


def test_record_ids_with_categories_only_deletes_requested(
    storage: LanceDBStorage,
) -> None:
    first = _record("first", categories=["note"])
    second = _record("second", categories=["note"])
    storage.save([first, second])

    deleted = storage.delete(record_ids=[first.id], categories=["note"])

    assert deleted == 1
    remaining = storage.search(first.embedding, limit=10)
    assert [record.id for record, _ in remaining] == ["second"]


def test_record_ids_with_metadata_filter_only_deletes_requested(
    storage: LanceDBStorage,
) -> None:
    first = _record("first", metadata={"kind": "pref"})
    second = _record("second", metadata={"kind": "pref"})
    storage.save([first, second])

    deleted = storage.delete(record_ids=[first.id], metadata_filter={"kind": "pref"})

    assert deleted == 1
    remaining = storage.search(first.embedding, limit=10)
    assert [record.id for record, _ in remaining] == ["second"]
