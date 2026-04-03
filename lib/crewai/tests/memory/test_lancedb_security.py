"""Tests for LanceDB storage security and edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def storage(tmp_path: Path):
    """Create a temporary LanceDB storage."""
    return LanceDBStorage(path=str(tmp_path / "test_db"), vector_dim=4)


def test_lancedb_scope_prefix_escaping(storage):
    """Test that malicious scope prefixes are properly escaped."""
    # Save a normal record
    normal = MemoryRecord(
        content="normal content",
        scope="/safe/path",
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    storage.save([normal])

    # Try to inject SQL with a malicious scope prefix
    malicious_prefix = "/safe' OR '1'='1"

    # This should NOT return all records due to SQL injection
    results = storage.search(
        [0.1, 0.2, 0.3, 0.4],
        scope_prefix=malicious_prefix,
        limit=10,
    )

    # Should return 0 results because the malicious prefix doesn't match
    assert len(results) == 0

    # Verify the normal query still works
    results = storage.search(
        [0.1, 0.2, 0.3, 0.4],
        scope_prefix="/safe",
        limit=10,
    )
    assert len(results) == 1


def test_lancedb_scope_prefix_escaping_in_delete(storage):
    """Test scope prefix escaping in delete operations."""
    # Save records in different scopes
    storage.save([
        MemoryRecord(content="a", scope="/safe/a", embedding=[0.1] * 4),
        MemoryRecord(content="b", scope="/safe/b", embedding=[0.2] * 4),
        MemoryRecord(content="c", scope="/other", embedding=[0.3] * 4),
    ])

    # Try to delete with malicious scope prefix
    malicious_prefix = "/safe' OR scope LIKE '%"
    deleted = storage.delete(scope_prefix=malicious_prefix)

    # Should delete 0 records (malicious prefix doesn't match anything)
    assert deleted == 0

    # All records should still exist
    assert storage.count() == 3


def test_lancedb_scope_prefix_escaping_in_scan(storage):
    """Test scope prefix escaping in _scan_rows."""
    storage.save([
        MemoryRecord(content="a", scope="/test/a", embedding=[0.1] * 4),
        MemoryRecord(content="b", scope="/test/b", embedding=[0.2] * 4),
    ])

    # Try to scan with malicious scope prefix
    malicious_prefix = "/test' OR '1'='1"
    records = storage.list_records(scope_prefix=malicious_prefix, limit=10)

    # Should return 0 records
    assert len(records) == 0

    # Normal scan should work
    records = storage.list_records(scope_prefix="/test", limit=10)
    assert len(records) == 2


def test_lancedb_concurrent_saves(storage):
    """Test that concurrent saves don't cause commit conflicts."""
    import threading

    def save_batch(batch_id: int):
        records = [
            MemoryRecord(
                content=f"batch {batch_id} item {i}",
                scope=f"/batch{batch_id}",
                embedding=[float(batch_id), float(i), 0.0, 0.0],
            )
            for i in range(5)
        ]
        storage.save(records)

    # Launch 3 concurrent save operations
    threads = [threading.Thread(target=save_batch, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 15 records should be saved
    assert storage.count() >= 15


def test_lancedb_empty_scope_prefix_handling(storage):
    """Test that empty/None scope prefixes are handled correctly."""
    storage.save([
        MemoryRecord(content="root", scope="/", embedding=[0.1] * 4),
        MemoryRecord(content="nested", scope="/a/b", embedding=[0.2] * 4),
    ])

    # None scope prefix should search all
    results = storage.search([0.1] * 4, scope_prefix=None, limit=10)
    assert len(results) == 2

    # Empty string should search all
    results = storage.search([0.1] * 4, scope_prefix="", limit=10)
    assert len(results) == 2

    # "/" should search all
    results = storage.search([0.1] * 4, scope_prefix="/", limit=10)
    assert len(results) == 2


def test_lancedb_special_characters_in_scope(storage):
    """Test that special characters in scope paths are handled correctly."""
    special_scopes = [
        "/path/with spaces",
        "/path/with-dashes",
        "/path/with_underscores",
        "/path/with.dots",
    ]

    for scope in special_scopes:
        record = MemoryRecord(
            content=f"content for {scope}",
            scope=scope,
            embedding=[0.1] * 4,
        )
        storage.save([record])

    # Verify all were saved
    assert storage.count() == len(special_scopes)

    # Verify each can be retrieved
    for scope in special_scopes:
        results = storage.search([0.1] * 4, scope_prefix=scope, limit=1)
        assert len(results) == 1
        assert results[0][0].scope == scope


def test_lancedb_update_preserves_created_at(storage):
    """Test that update preserves created_at timestamp."""
    from datetime import datetime, timedelta

    # Save a record
    original = MemoryRecord(
        content="original content",
        scope="/test",
        embedding=[0.1] * 4,
    )
    storage.save([original])

    # Get the record back
    results = storage.search([0.1] * 4, scope_prefix="/test", limit=1)
    saved = results[0][0]
    original_created_at = saved.created_at

    # Wait a bit and update
    import time
    time.sleep(0.1)

    updated = MemoryRecord(
        id=saved.id,
        content="updated content",
        scope=saved.scope,
        categories=saved.categories,
        metadata=saved.metadata,
        importance=saved.importance,
        created_at=saved.created_at,
        last_accessed=datetime.utcnow(),
        embedding=[0.2] * 4,
    )
    storage.update(updated)

    # Retrieve again
    results = storage.search([0.2] * 4, scope_prefix="/test", limit=1)
    final = results[0][0]

    # created_at should be preserved
    assert final.created_at == original_created_at
    # content should be updated
    assert final.content == "updated content"
    # last_accessed should be newer
    assert final.last_accessed > original_created_at
