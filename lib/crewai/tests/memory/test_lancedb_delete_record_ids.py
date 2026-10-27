"""Regression tests for record_ids intersection in LanceDBStorage.delete().

record_ids must intersect with the other filters (older_than, categories,
metadata_filter), matching QdrantEdgeStorage's semantics, instead of being
ignored by the filtered paths. The scoped cases also pin down the WHERE-clause
parenthesization: the scope predicate is an OR group, so an unparenthesized
join let its root-scope branch escape the other filters.
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


def test_record_ids_with_scope_prefix_and_older_than_keeps_other_records(
    storage: LanceDBStorage,
) -> None:
    """record_ids and older_than must also constrain the root-scope branch.

    The scope predicate is an OR group (``... OR scope = '/'``) and AND binds
    tighter than OR, so joining the conditions unparenthesized produced
    ``(id IN (...) AND scope LIKE '/proj%') OR (scope = '/' AND created_at <
    t)``: a scoped delete naming one id also removed every root-scope record
    older than the threshold, and skipped the age check for the named ids
    inside the prefix.
    """
    in_old = _record("in-old", scope="/proj/a")
    in_old.created_at = datetime.utcnow() - timedelta(days=30)
    in_new = _record("in-new", scope="/proj/b")
    root_old = _record("root-old", scope="/")
    root_old.created_at = datetime.utcnow() - timedelta(days=30)
    storage.save([in_old, in_new, root_old])

    deleted = storage.delete(
        record_ids=[in_old.id, in_new.id],
        scope_prefix="/proj",
        older_than=datetime.utcnow() - timedelta(days=1),
    )

    assert deleted == 1
    remaining = storage.search(in_old.embedding, limit=10)
    assert sorted(record.id for record, _ in remaining) == ["in-new", "root-old"]


def test_scope_prefix_with_older_than_honours_the_age_filter(
    storage: LanceDBStorage,
) -> None:
    """The same precedence defect, reachable without record_ids at all.

    Unparenthesized, the age filter only applied to the ``scope = '/'`` branch,
    so a scoped delete removed every record under the prefix however new it
    was. Pre-existing rather than introduced here, and fixed by the same
    parenthesization.
    """
    old = _record("old", scope="/proj/a")
    old.created_at = datetime.utcnow() - timedelta(days=30)
    fresh = _record("fresh", scope="/proj/b")
    storage.save([old, fresh])

    deleted = storage.delete(
        scope_prefix="/proj",
        older_than=datetime.utcnow() - timedelta(days=1),
    )

    assert deleted == 1
    remaining = storage.search(old.embedding, limit=10)
    assert [record.id for record, _ in remaining] == ["fresh"]
