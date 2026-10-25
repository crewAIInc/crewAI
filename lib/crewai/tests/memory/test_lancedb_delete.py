"""Tests for LanceDBStorage delete with various older_than configurations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def lancedb_storage(tmp_path: Path) -> LanceDBStorage:
    return LanceDBStorage(path=tmp_path / "memory", vector_dim=4, compact_every=0)


def _make_record(
    record_id: str,
    created_at: datetime,
    categories: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"Record {record_id}",
        scope="/",
        categories=categories or ["agent"],
        embedding=[0.1, 0.2, 0.3, 0.4],
        created_at=created_at,
    )


@pytest.mark.parametrize(
    ("older_than", "expect_deleted_id"),
    [
        (None, None),
        (datetime(2025, 1, 2, 12, 0, 0), "rec_old"),
        (datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc), "rec_old"),
        (
            datetime(2025, 1, 2, 17, 0, 0, tzinfo=timezone(timedelta(hours=5))),
            "rec_old",
        ),
        (
            datetime(2025, 1, 2, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5))),
            "rec_old",
        ),
    ],
    ids=["none", "naive", "utc", "non_utc_ahead", "non_utc_behind"],
)
def test_lancedb_delete_older_than_with_categories(
    lancedb_storage: LanceDBStorage,
    older_than: datetime | None,
    expect_deleted_id: str | None,
) -> None:
    old_rec = _make_record("rec_old", datetime(2025, 1, 1, 12, 0, 0), ["agent"])
    new_rec = _make_record("rec_new", datetime(2025, 1, 3, 12, 0, 0), ["agent"])
    lancedb_storage.save([old_rec, new_rec])

    deleted_count = lancedb_storage.delete(categories=["agent"], older_than=older_than)

    if older_than is None:
        assert deleted_count == 2
        assert lancedb_storage.get_record("rec_old") is None
        assert lancedb_storage.get_record("rec_new") is None
    else:
        assert deleted_count == 1
        assert lancedb_storage.get_record("rec_old") is None
        assert lancedb_storage.get_record("rec_new") is not None


@pytest.mark.parametrize(
    ("older_than", "expect_deleted_id"),
    [
        (None, None),
        (datetime(2025, 1, 2, 12, 0, 0), "rec_old"),
        (datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc), "rec_old"),
        (
            datetime(2025, 1, 2, 17, 0, 0, tzinfo=timezone(timedelta(hours=5))),
            "rec_old",
        ),
        (
            datetime(2025, 1, 2, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5))),
            "rec_old",
        ),
    ],
    ids=["none", "naive", "utc", "non_utc_ahead", "non_utc_behind"],
)
def test_lancedb_delete_older_than_without_categories(
    lancedb_storage: LanceDBStorage,
    older_than: datetime | None,
    expect_deleted_id: str | None,
) -> None:
    old_rec = _make_record("rec_old", datetime(2025, 1, 1, 12, 0, 0))
    new_rec = _make_record("rec_new", datetime(2025, 1, 3, 12, 0, 0))
    lancedb_storage.save([old_rec, new_rec])

    deleted_count = lancedb_storage.delete(older_than=older_than)

    if older_than is None:
        assert deleted_count == 2
        assert lancedb_storage.get_record("rec_old") is None
        assert lancedb_storage.get_record("rec_new") is None
    else:
        assert deleted_count == 1
        assert lancedb_storage.get_record("rec_old") is None
        assert lancedb_storage.get_record("rec_new") is not None
