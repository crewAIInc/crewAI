"""Unit tests for LanceDBStorage.delete filtering and record_ids intersection."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
import pytest


DIM = 4


def _record(
    record_id: str,
    scope: str = "/test",
    categories: list[str] | None = None,
    created_at: datetime | None = None,
    metadata: dict[str, object] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        categories=categories or ["default"],
        created_at=created_at or datetime.now(timezone.utc),
        metadata=metadata or {},
        embedding=[0.1] * DIM,
    )


@pytest.fixture
def storage(tmp_path: Path) -> LanceDBStorage:
    return LanceDBStorage(path=tmp_path / "memory", vector_dim=DIM, compact_every=0)


def test_delete_with_only_record_ids(storage: LanceDBStorage) -> None:
    records = [_record("r1"), _record("r2"), _record("r3")]
    storage.save(records)

    deleted = storage.delete(record_ids=["r1", "r2"])
    assert deleted == 2

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r3"}


def test_delete_with_record_ids_and_older_than(storage: LanceDBStorage) -> None:
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=10)
    new_time = now

    r1 = _record("r1", created_at=old_time)
    r2 = _record("r2", created_at=new_time)
    r3 = _record("r3", created_at=old_time)
    storage.save([r1, r2, r3])

    cutoff = now - timedelta(days=5)
    # Target r1 and r2; only r1 is older than cutoff, r2 is newer
    deleted = storage.delete(record_ids=["r1", "r2"], older_than=cutoff)
    assert deleted == 1

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r2", "r3"}


def test_delete_with_record_ids_and_categories(storage: LanceDBStorage) -> None:
    r1 = _record("r1", categories=["work", "urgent"])
    r2 = _record("r2", categories=["personal"])
    r3 = _record("r3", categories=["work"])
    storage.save([r1, r2, r3])

    # Target r1 and r2 with category "work"; only r1 matches category
    deleted = storage.delete(record_ids=["r1", "r2"], categories=["work"])
    assert deleted == 1

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r2", "r3"}


def test_records_outside_record_ids_not_deleted_even_if_matching_older_than(
    storage: LanceDBStorage,
) -> None:
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=20)

    # All three records are old, but only r1 is in record_ids
    r1 = _record("r1", created_at=old_time)
    r2 = _record("r2", created_at=old_time)
    r3 = _record("r3", created_at=old_time)
    storage.save([r1, r2, r3])

    cutoff = now - timedelta(days=5)
    deleted = storage.delete(record_ids=["r1"], older_than=cutoff)
    assert deleted == 1

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r2", "r3"}


def test_records_outside_record_ids_not_deleted_even_if_matching_categories(
    storage: LanceDBStorage,
) -> None:
    r1 = _record("r1", categories=["archive"])
    r2 = _record("r2", categories=["archive"])
    r3 = _record("r3", categories=["archive"])
    storage.save([r1, r2, r3])

    # All match "archive", but only r1 and r2 are in record_ids
    deleted = storage.delete(record_ids=["r1", "r2"], categories=["archive"])
    assert deleted == 2

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r3"}


def test_delete_with_record_ids_categories_and_older_than_intersection(
    storage: LanceDBStorage,
) -> None:
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=10)
    new_time = now

    r1 = _record("r1", categories=["note"], created_at=old_time)  # matches all
    r2 = _record("r2", categories=["note"], created_at=new_time)  # in ids and cat, but too new
    r3 = _record("r3", categories=["other"], created_at=old_time)  # in ids and old, wrong cat
    r4 = _record("r4", categories=["note"], created_at=old_time)  # matches cat and old, not in ids
    storage.save([r1, r2, r3, r4])

    cutoff = now - timedelta(days=5)
    deleted = storage.delete(
        record_ids=["r1", "r2", "r3"],
        categories=["note"],
        older_than=cutoff,
    )
    assert deleted == 1

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r2", "r3", "r4"}


def test_delete_with_record_ids_and_metadata_filter(storage: LanceDBStorage) -> None:
    r1 = _record("r1", metadata={"status": "draft", "user_id": 10})
    r2 = _record("r2", metadata={"status": "published", "user_id": 10})
    r3 = _record("r3", metadata={"status": "draft", "user_id": 20})
    storage.save([r1, r2, r3])

    deleted = storage.delete(
        record_ids=["r1", "r2"],
        metadata_filter={"status": "draft"},
    )
    assert deleted == 1

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r2", "r3"}


def test_delete_with_record_ids_and_scope_prefix(storage: LanceDBStorage) -> None:
    r1 = _record("r1", scope="/agents/researcher")
    r2 = _record("r2", scope="/agents/writer")
    r3 = _record("r3", scope="/agents/researcher")
    storage.save([r1, r2, r3])

    deleted = storage.delete(
        record_ids=["r1", "r2"],
        scope_prefix="/agents/researcher",
    )
    assert deleted == 1

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r2", "r3"}


def test_delete_with_empty_record_ids_is_noop(storage: LanceDBStorage) -> None:
    r1 = _record("r1")
    r2 = _record("r2")
    storage.save([r1, r2])

    deleted = storage.delete(record_ids=[])
    assert deleted == 0

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r1", "r2"}


def test_delete_with_nonexistent_record_ids(storage: LanceDBStorage) -> None:
    r1 = _record("r1")
    storage.save([r1])

    deleted = storage.delete(record_ids=["missing-1", "missing-2"])
    assert deleted == 0

    remaining_ids = {r.id for r in storage.list_records(limit=100)}
    assert remaining_ids == {"r1"}
