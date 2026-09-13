"""Tests for LanceDBStorage.delete() filter isolation fixes (issue #7419)."""

import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def storage():
    """Create a temporary LanceDB storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = LanceDBStorage(path=tmpdir, vector_dim=4)
        yield s


def _make_record(
    record_id: str,
    scope: str = "/",
    categories: list[str] | None = None,
    created_at: datetime | None = None,
    metadata: dict | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content for {record_id}",
        scope=scope,
        categories=categories or [],
        created_at=created_at or datetime.now(timezone.utc),
        embedding=[0.1] * 4,
        metadata=metadata or {},
    )


class TestDeleteScopeIsolation:
    """record_ids must respect scope_prefix -- never cross-scope delete."""

    def test_scope_prefix_restricts_record_id_deletion(self, storage):
        """Deleting record_ids with a scope only deletes IDs within that scope."""
        now = datetime.now(timezone.utc)
        r1 = _make_record("rec1", scope="/tenantA", created_at=now)
        r2 = _make_record("rec2", scope="/tenantB", created_at=now)
        storage.save([r1, r2])

        deleted = storage.delete(scope_prefix="/tenantA", record_ids=["rec2"])
        assert deleted == 0
        assert storage.get_record("rec2") is not None

    def test_scope_prefix_allows_matching_record_id_deletion(self, storage):
        """Deleting record_ids with matching scope deletes the record."""
        now = datetime.now(timezone.utc)
        r1 = _make_record("rec1", scope="/tenantA", created_at=now)
        r2 = _make_record("rec2", scope="/tenantB", created_at=now)
        storage.save([r1, r2])

        deleted = storage.delete(scope_prefix="/tenantA", record_ids=["rec1"])
        assert deleted == 1
        assert storage.get_record("rec1") is None
        assert storage.get_record("rec2") is not None


class TestDeleteRecordIdsWithCategories:
    """When both record_ids and categories are given, only IDs in the set AND
    matching the categories should be deleted."""

    def test_categories_does_not_cause_mass_deletion(self, storage):
        """Providing categories with record_ids should NOT delete all category matches."""
        now = datetime.now(timezone.utc)
        r1 = _make_record("rec1", categories=["catA"], created_at=now)
        r2 = _make_record("rec2", categories=["catA"], created_at=now)
        r3 = _make_record("rec3", categories=["catA"], created_at=now)
        storage.save([r1, r2, r3])

        deleted = storage.delete(record_ids=["rec1"], categories=["catA"])
        assert deleted == 1
        assert storage.get_record("rec1") is None
        assert storage.get_record("rec2") is not None
        assert storage.get_record("rec3") is not None


class TestDeleteOlderThan:
    """older_than filter must be respected even when record_ids are given."""

    def test_older_than_prevents_recent_deletion(self, storage):
        """recently created records should NOT be deleted when older_than is set."""
        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=1)
        old = now - timedelta(days=10)
        r1 = _make_record("rec1", created_at=recent)
        r2 = _make_record("rec2", created_at=old)
        storage.save([r1, r2])

        thirty_days_ago = now - timedelta(days=30)
        deleted = storage.delete(record_ids=["rec1", "rec2"], older_than=thirty_days_ago)
        # rec1 is 1h old, not older than 30 days -- should NOT be deleted
        # rec2 is 10 days old, not older than 30 days -- should NOT be deleted
        assert deleted == 0
        assert storage.get_record("rec1") is not None
        assert storage.get_record("rec2") is not None

    def test_older_than_allows_old_record_deletion(self, storage):
        """Records older than the threshold SHOULD be deleted."""
        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=1)
        old = now - timedelta(days=10)
        r1 = _make_record("rec1", created_at=recent)
        r2 = _make_record("rec2", created_at=old)
        storage.save([r1, r2])

        five_days_ago = now - timedelta(days=5)
        deleted = storage.delete(record_ids=["rec1", "rec2"], older_than=five_days_ago)
        # rec1 is 1h old -- not older than 5 days
        # rec2 is 10 days old -- older than 5 days
        assert deleted == 1
        assert storage.get_record("rec1") is not None
        assert storage.get_record("rec2") is None
