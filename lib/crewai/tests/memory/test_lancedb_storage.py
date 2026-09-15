"""Tests for LanceDBStorage backend."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


def test_list_records_returns_newest_first_with_limit(tmp_path: Path) -> None:
    """list_records(limit=N) should return the N newest records, not the N oldest."""
    storage = LanceDBStorage(path=str(tmp_path), vector_dim=4)
    base_time = datetime.now(timezone.utc)

    # Insert 10 records with sequential created_at timestamps:
    # rec_0 is oldest, rec_9 is newest
    records = [
        MemoryRecord(
            id=f"rec_{i}",
            content=f"Record {i}",
            scope="/test",
            created_at=base_time + timedelta(minutes=i),
            updated_at=base_time + timedelta(minutes=i),
            vector=[0.1, 0.2, 0.3, 0.4],
        )
        for i in range(10)
    ]
    storage.save(records)

    # Request the 3 newest records
    results = storage.list_records(scope_prefix="/test", limit=3)
    result_ids = [r.id for r in results]

    assert result_ids == ["rec_9", "rec_8", "rec_7"], (
        f"Expected newest records ['rec_9', 'rec_8', 'rec_7'], but got {result_ids}"
    )


def test_list_records_pagination_with_offset(tmp_path: Path) -> None:
    """list_records with limit and offset should paginate over newest records first."""
    storage = LanceDBStorage(path=str(tmp_path), vector_dim=4)
    base_time = datetime.now(timezone.utc)

    records = [
        MemoryRecord(
            id=f"rec_{i}",
            content=f"Record {i}",
            scope="/test",
            created_at=base_time + timedelta(minutes=i),
            updated_at=base_time + timedelta(minutes=i),
            vector=[0.1, 0.2, 0.3, 0.4],
        )
        for i in range(10)
    ]
    storage.save(records)

    # First page: newest 3
    page1 = storage.list_records(scope_prefix="/test", limit=3, offset=0)
    assert [r.id for r in page1] == ["rec_9", "rec_8", "rec_7"]

    # Second page: next 3
    page2 = storage.list_records(scope_prefix="/test", limit=3, offset=3)
    assert [r.id for r in page2] == ["rec_6", "rec_5", "rec_4"]
