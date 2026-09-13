"""Tests for LanceDBStorage.list_records() ordering fix (issue #7394)."""

import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LanceDBStorage(path=tmpdir, vector_dim=4)


def _make_record(record_id: str, created_at: datetime) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content for {record_id}",
        created_at=created_at,
        embedding=[0.1] * 4,
    )


class TestListRecordsOrdering:
    """list_records must return newest records first."""

    def test_returns_newest_first(self, storage):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        records = [
            _make_record(f"rec_{i}", created_at=base + timedelta(minutes=i))
            for i in range(10)
        ]
        storage.save(records)

        results = storage.list_records(limit=3)
        ids = [r.id for r in results]
        assert ids == ["rec_9", "rec_8", "rec_7"]

    def test_offset_skips_newest(self, storage):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        records = [
            _make_record(f"rec_{i}", created_at=base + timedelta(minutes=i))
            for i in range(10)
        ]
        storage.save(records)

        results = storage.list_records(limit=3, offset=3)
        ids = [r.id for r in results]
        # After skipping 3 newest, expect rec_6, rec_5, rec_4
        assert ids == ["rec_6", "rec_5", "rec_4"]

    def test_empty_scope_returns_empty(self, storage):
        results = storage.list_records(scope_prefix="/nonexistent")
        assert results == []
