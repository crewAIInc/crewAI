"""Tests for ValkeyStorage scope operations."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from crewai.memory.storage.valkey_storage import ValkeyStorage
from crewai.memory.types import MemoryRecord, ScopeInfo


@pytest.fixture
def mock_glide_client() -> AsyncMock:
    """Create a mock GlideClient for testing."""
    client = AsyncMock()
    client.hset = AsyncMock(return_value=1)
    client.zrange = AsyncMock(return_value=[])
    client.zadd = AsyncMock()
    client.sadd = AsyncMock()
    client.zrem = AsyncMock()
    client.srem = AsyncMock()
    client.hgetall = AsyncMock(return_value={})
    client.scan = AsyncMock()
    client.smembers = AsyncMock(return_value=[])
    client.scard = AsyncMock(return_value=0)
    client.delete = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def valkey_storage(mock_glide_client: AsyncMock) -> ValkeyStorage:
    """Create a ValkeyStorage instance with mocked client."""
    storage = ValkeyStorage(host="localhost", port=6379, db=0)

    # Mock the client creation to return our mock
    async def mock_create_client() -> AsyncMock:
        storage._client = mock_glide_client
        return mock_glide_client

    storage._get_client = mock_create_client  # type: ignore[method-assign]
    return storage


class TestValkeyStorageListRecords:
    """Tests for list_records operation."""

    @pytest.mark.asyncio
    async def test_list_records_returns_newest_first(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that list_records returns records ordered by created_at descending."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",  # cursor
            [b"scope:/test"],  # keys
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2", b"record-3"],  # ZRANGE response
        ]

        # Mock hgetall to return record data
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            if key == "record:record-1":
                return {
                    b"id": b"record-1",
                    b"content": b"Content 1",
                    b"scope": b"/test",
                    b"categories": b"[]",
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-01T10:00:00",
                    b"last_accessed": b"2024-01-01T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-2":
                return {
                    b"id": b"record-2",
                    b"content": b"Content 2",
                    b"scope": b"/test",
                    b"categories": b"[]",
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-02T10:00:00",
                    b"last_accessed": b"2024-01-02T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-3":
                return {
                    b"id": b"record-3",
                    b"content": b"Content 3",
                    b"scope": b"/test",
                    b"categories": b"[]",
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-03T10:00:00",
                    b"last_accessed": b"2024-01-03T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            return {}

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # List records
        records = await valkey_storage._alist_records(scope_prefix="/test")

        # Verify records are ordered newest first
        assert len(records) == 3
        assert records[0].id == "record-3"  # Newest
        assert records[1].id == "record-2"
        assert records[2].id == "record-1"  # Oldest

    @pytest.mark.asyncio
    async def test_list_records_with_pagination_limit_only(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test pagination with limit only (no offset)."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2", b"record-3", b"record-4", b"record-5"],
        ]

        # Mock hgetall to return record data
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            record_id = key.split(":")[-1]
            day = int(record_id.split("-")[-1])
            return {
                b"id": record_id.encode(),
                b"content": f"Content {day}".encode(),
                b"scope": b"/test",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": f"2024-01-0{day}T10:00:00".encode(),
                b"last_accessed": f"2024-01-0{day}T10:00:00".encode(),
                b"source": b"",
                b"private": b"false",
                b"embedding": b"",
            }

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # List records with limit only
        records = await valkey_storage._alist_records(scope_prefix="/test", limit=3)

        # Verify limit works (take first 3)
        assert len(records) == 3
        assert records[0].id == "record-5"  # Newest
        assert records[1].id == "record-4"
        assert records[2].id == "record-3"

    @pytest.mark.asyncio
    async def test_list_records_with_pagination_offset_only(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test pagination with offset only (default limit)."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2", b"record-3"],
        ]

        # Mock hgetall to return record data
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            record_id = key.split(":")[-1]
            day = int(record_id.split("-")[-1])
            return {
                b"id": record_id.encode(),
                b"content": f"Content {day}".encode(),
                b"scope": b"/test",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": f"2024-01-0{day}T10:00:00".encode(),
                b"last_accessed": f"2024-01-0{day}T10:00:00".encode(),
                b"source": b"",
                b"private": b"false",
                b"embedding": b"",
            }

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # List records with offset only
        records = await valkey_storage._alist_records(scope_prefix="/test", offset=1)

        # Verify offset works (skip first 1)
        assert len(records) == 2
        assert records[0].id == "record-2"
        assert records[1].id == "record-1"

    @pytest.mark.asyncio
    async def test_list_records_with_pagination_limit_and_offset(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test pagination with both limit and offset."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2", b"record-3", b"record-4", b"record-5"],
        ]

        # Mock hgetall to return record data
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            record_id = key.split(":")[-1]
            day = int(record_id.split("-")[-1])
            return {
                b"id": record_id.encode(),
                b"content": f"Content {day}".encode(),
                b"scope": b"/test",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": f"2024-01-0{day}T10:00:00".encode(),
                b"last_accessed": f"2024-01-0{day}T10:00:00".encode(),
                b"source": b"",
                b"private": b"false",
                b"embedding": b"",
            }

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # List records with pagination
        records = await valkey_storage._alist_records(
            scope_prefix="/test", limit=2, offset=1
        )

        # Verify pagination works (skip 1, take 2)
        assert len(records) == 2
        assert records[0].id == "record-4"  # Second newest
        assert records[1].id == "record-3"  # Third newest

    @pytest.mark.asyncio
    async def test_list_records_with_large_offset(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test pagination with offset beyond available records."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2"],
        ]

        # Mock hgetall to return record data
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            record_id = key.split(":")[-1]
            day = int(record_id.split("-")[-1])
            return {
                b"id": record_id.encode(),
                b"content": f"Content {day}".encode(),
                b"scope": b"/test",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": f"2024-01-0{day}T10:00:00".encode(),
                b"last_accessed": f"2024-01-0{day}T10:00:00".encode(),
                b"source": b"",
                b"private": b"false",
                b"embedding": b"",
            }

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # List records with large offset
        records = await valkey_storage._alist_records(scope_prefix="/test", offset=10)

        # Verify empty list when offset exceeds available records
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_list_records_empty_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_records returns empty list for empty scope."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/empty"],
        )

        # Mock ZRANGE to return no record IDs
        mock_glide_client.zrange.side_effect = [
            [],  # No records
        ]

        # List records
        records = await valkey_storage._alist_records(scope_prefix="/empty")

        # Verify empty list
        assert len(records) == 0

    def test_list_records_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync list_records wrapper calls async implementation."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1"],
        ]

        # Mock hgetall to return record data
        mock_glide_client.hgetall.return_value = {
            b"id": b"record-1",
            b"content": b"Content 1",
            b"scope": b"/test",
            b"categories": b"[]",
            b"metadata": b"{}",
            b"importance": b"0.5",
            b"created_at": b"2024-01-01T10:00:00",
            b"last_accessed": b"2024-01-01T10:00:00",
            b"source": b"",
            b"private": b"false",
            b"embedding": b"",
        }

        # Call sync wrapper
        records = valkey_storage.list_records(scope_prefix="/test")

        # Verify it works
        assert len(records) == 1
        assert records[0].id == "record-1"


class TestValkeyStorageGetScopeInfo:
    """Tests for get_scope_info operation."""

    @pytest.mark.asyncio
    async def test_get_scope_info_returns_accurate_counts(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that get_scope_info returns accurate record counts and metadata."""
        # Mock scan to return scope keys
        mock_glide_client.scan.side_effect = [
            (b"0", [b"scope:/test", b"scope:/test/sub"]),  # First scan
            (b"0", [b"scope:/test", b"scope:/test/sub"]),  # Second scan for child scopes
        ]

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2"],  # Records in /test
            [b"record-3"],  # Records in /test/sub
        ]

        # Mock hgetall to return record data
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            if key == "record:record-1":
                return {
                    b"id": b"record-1",
                    b"content": b"Content 1",
                    b"scope": b"/test",
                    b"categories": b'["planning"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-01T10:00:00",
                    b"last_accessed": b"2024-01-01T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-2":
                return {
                    b"id": b"record-2",
                    b"content": b"Content 2",
                    b"scope": b"/test",
                    b"categories": b'["execution"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-03T10:00:00",
                    b"last_accessed": b"2024-01-03T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-3":
                return {
                    b"id": b"record-3",
                    b"content": b"Content 3",
                    b"scope": b"/test/sub",
                    b"categories": b'["planning"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-02T10:00:00",
                    b"last_accessed": b"2024-01-02T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            return {}

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # Get scope info
        info = await valkey_storage._aget_scope_info("/test")

        # Verify scope info
        assert info.path == "/test"
        assert info.record_count == 3  # All records in /test and subscopes
        assert set(info.categories) == {"execution", "planning"}
        assert info.oldest_record == datetime(2024, 1, 1, 10, 0, 0)
        assert info.newest_record == datetime(2024, 1, 3, 10, 0, 0)
        assert "/test/sub" in info.child_scopes

    @pytest.mark.asyncio
    async def test_get_scope_info_returns_accurate_timestamps(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that get_scope_info returns accurate oldest and newest timestamps."""
        # Mock scan to return scope keys
        mock_glide_client.scan.side_effect = [
            (b"0", [b"scope:/test"]),  # First scan
            (b"0", [b"scope:/test"]),  # Second scan for child scopes
        ]

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2", b"record-3"],
        ]

        # Mock hgetall to return record data with different timestamps
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            if key == "record:record-1":
                return {
                    b"id": b"record-1",
                    b"content": b"Content 1",
                    b"scope": b"/test",
                    b"categories": b"[]",
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-15T10:00:00",
                    b"last_accessed": b"2024-01-15T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-2":
                return {
                    b"id": b"record-2",
                    b"content": b"Content 2",
                    b"scope": b"/test",
                    b"categories": b"[]",
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-01T10:00:00",  # Oldest
                    b"last_accessed": b"2024-01-01T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-3":
                return {
                    b"id": b"record-3",
                    b"content": b"Content 3",
                    b"scope": b"/test",
                    b"categories": b"[]",
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-20T10:00:00",  # Newest
                    b"last_accessed": b"2024-01-20T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            return {}

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # Get scope info
        info = await valkey_storage._aget_scope_info("/test")

        # Verify timestamps
        assert info.oldest_record == datetime(2024, 1, 1, 10, 0, 0)
        assert info.newest_record == datetime(2024, 1, 20, 10, 0, 0)

    @pytest.mark.asyncio
    async def test_get_scope_info_empty_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test get_scope_info returns empty info for empty scope."""
        # Mock scan to return no matching scopes
        mock_glide_client.scan.return_value = (b"0", [])

        # Get scope info for empty scope
        info = await valkey_storage._aget_scope_info("/empty")

        # Verify empty scope info
        assert info.path == "/empty"
        assert info.record_count == 0
        assert info.categories == []
        assert info.oldest_record is None
        assert info.newest_record is None
        assert info.child_scopes == []

    @pytest.mark.asyncio
    async def test_get_scope_info_with_multiple_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test get_scope_info aggregates categories from all records."""
        # Mock scan to return scope keys
        mock_glide_client.scan.side_effect = [
            (b"0", [b"scope:/test"]),  # First scan
            (b"0", [b"scope:/test"]),  # Second scan for child scopes
        ]

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2", b"record-3"],
        ]

        # Mock hgetall to return record data with various categories
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            if key == "record:record-1":
                return {
                    b"id": b"record-1",
                    b"content": b"Content 1",
                    b"scope": b"/test",
                    b"categories": b'["planning", "execution"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-01T10:00:00",
                    b"last_accessed": b"2024-01-01T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-2":
                return {
                    b"id": b"record-2",
                    b"content": b"Content 2",
                    b"scope": b"/test",
                    b"categories": b'["review", "planning"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-02T10:00:00",
                    b"last_accessed": b"2024-01-02T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-3":
                return {
                    b"id": b"record-3",
                    b"content": b"Content 3",
                    b"scope": b"/test",
                    b"categories": b'["analysis"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-03T10:00:00",
                    b"last_accessed": b"2024-01-03T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            return {}

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # Get scope info
        info = await valkey_storage._aget_scope_info("/test")

        # Verify all unique categories are collected and sorted
        assert set(info.categories) == {"analysis", "execution", "planning", "review"}
        assert info.categories == ["analysis", "execution", "planning", "review"]  # Sorted

    def test_get_scope_info_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync get_scope_info wrapper calls async implementation."""
        # Mock scan to return no matching scopes
        mock_glide_client.scan.return_value = (b"0", [])

        # Call sync wrapper
        info = valkey_storage.get_scope_info("/test")

        # Verify it works
        assert info.path == "/test"
        assert info.record_count == 0


class TestValkeyStorageListScopes:
    """Tests for list_scopes operation."""

    @pytest.mark.asyncio
    async def test_list_scopes_returns_immediate_children_only(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that list_scopes returns only immediate children, not grandchildren."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [
                b"scope:/agent",
                b"scope:/agent/task",
                b"scope:/agent/task/subtask",
                b"scope:/crew",
            ],
        )

        # List scopes under root
        scopes = await valkey_storage._alist_scopes("/")

        # Verify only immediate children are returned
        assert len(scopes) == 2
        assert "/agent" in scopes
        assert "/crew" in scopes
        assert "/agent/task" not in scopes  # Grandchild not included

    @pytest.mark.asyncio
    async def test_list_scopes_with_parent(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_scopes with specific parent path."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [
                b"scope:/agent",
                b"scope:/agent/task",
                b"scope:/agent/task/subtask",
                b"scope:/agent/memory",
            ],
        )

        # List scopes under /agent
        scopes = await valkey_storage._alist_scopes("/agent")

        # Verify only immediate children of /agent are returned
        assert len(scopes) == 2
        assert "/agent/task" in scopes
        assert "/agent/memory" in scopes
        assert "/agent/task/subtask" not in scopes  # Grandchild not included

    @pytest.mark.asyncio
    async def test_list_scopes_returns_sorted_order(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that list_scopes returns scopes in sorted order."""
        # Mock scan to return scope keys in random order
        mock_glide_client.scan.return_value = (
            b"0",
            [
                b"scope:/zebra",
                b"scope:/alpha",
                b"scope:/beta",
                b"scope:/gamma",
            ],
        )

        # List scopes under root
        scopes = await valkey_storage._alist_scopes("/")

        # Verify scopes are sorted
        assert scopes == ["/alpha", "/beta", "/gamma", "/zebra"]

    @pytest.mark.asyncio
    async def test_list_scopes_empty_parent(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_scopes returns empty list when parent has no children."""
        # Mock scan to return scope keys that don't match parent
        mock_glide_client.scan.return_value = (
            b"0",
            [
                b"scope:/agent",
                b"scope:/crew",
            ],
        )

        # List scopes under /other (no children)
        scopes = await valkey_storage._alist_scopes("/other")

        # Verify empty list
        assert len(scopes) == 0

    @pytest.mark.asyncio
    async def test_list_scopes_with_deep_hierarchy(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_scopes with deep scope hierarchy."""
        # Mock scan to return scope keys with deep nesting
        mock_glide_client.scan.return_value = (
            b"0",
            [
                b"scope:/a",
                b"scope:/a/b",
                b"scope:/a/b/c",
                b"scope:/a/b/c/d",
                b"scope:/a/x",
            ],
        )

        # List scopes under /a/b
        scopes = await valkey_storage._alist_scopes("/a/b")

        # Verify only immediate children are returned
        assert len(scopes) == 1
        assert "/a/b/c" in scopes
        assert "/a/b/c/d" not in scopes  # Grandchild not included

    def test_list_scopes_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync list_scopes wrapper calls async implementation."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/agent", b"scope:/crew"],
        )

        # Call sync wrapper
        scopes = valkey_storage.list_scopes("/")

        # Verify it works
        assert len(scopes) == 2
        assert "/agent" in scopes
        assert "/crew" in scopes


class TestValkeyStorageListCategories:
    """Tests for list_categories operation."""

    @pytest.mark.asyncio
    async def test_list_categories_global_returns_accurate_counts(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_categories returns accurate global category counts."""
        # Mock scan to return category keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"category:planning", b"category:execution", b"category:review"],
        )

        # Mock scard to return category counts
        def mock_scard(key: str) -> int:
            if key == "category:planning":
                return 5
            elif key == "category:execution":
                return 3
            elif key == "category:review":
                return 2
            return 0

        mock_glide_client.scard.side_effect = mock_scard

        # List categories globally
        categories = await valkey_storage._alist_categories(scope_prefix=None)

        # Verify category counts
        assert categories == {"planning": 5, "execution": 3, "review": 2}

    @pytest.mark.asyncio
    async def test_list_categories_with_scope_returns_accurate_counts(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_categories with scope filtering returns accurate counts."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2", b"record-3"],
        ]

        # Mock hgetall to return record data with categories
        def mock_hgetall(key: str) -> dict[bytes, bytes]:
            if key == "record:record-1":
                return {
                    b"id": b"record-1",
                    b"content": b"Content 1",
                    b"scope": b"/test",
                    b"categories": b'["planning", "execution"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-01T10:00:00",
                    b"last_accessed": b"2024-01-01T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-2":
                return {
                    b"id": b"record-2",
                    b"content": b"Content 2",
                    b"scope": b"/test",
                    b"categories": b'["planning"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-02T10:00:00",
                    b"last_accessed": b"2024-01-02T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            elif key == "record:record-3":
                return {
                    b"id": b"record-3",
                    b"content": b"Content 3",
                    b"scope": b"/test",
                    b"categories": b'["execution"]',
                    b"metadata": b"{}",
                    b"importance": b"0.5",
                    b"created_at": b"2024-01-03T10:00:00",
                    b"last_accessed": b"2024-01-03T10:00:00",
                    b"source": b"",
                    b"private": b"false",
                    b"embedding": b"",
                }
            return {}

        mock_glide_client.hgetall.side_effect = mock_hgetall

        # List categories in scope
        categories = await valkey_storage._alist_categories(scope_prefix="/test")

        # Verify category counts
        assert categories == {"planning": 2, "execution": 2}

    @pytest.mark.asyncio
    async def test_list_categories_empty_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_categories returns empty dict for empty scope."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/empty"],
        )

        # Mock ZRANGE to return no record IDs
        mock_glide_client.zrange.side_effect = [
            [],  # No records
        ]

        # List categories in empty scope
        categories = await valkey_storage._alist_categories(scope_prefix="/empty")

        # Verify empty dict
        assert categories == {}

    @pytest.mark.asyncio
    async def test_list_categories_global_empty(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test list_categories returns empty dict when no categories exist."""
        # Mock scan to return no category keys
        mock_glide_client.scan.return_value = (b"0", [])

        # List categories globally
        categories = await valkey_storage._alist_categories(scope_prefix=None)

        # Verify empty dict
        assert categories == {}

    def test_list_categories_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync list_categories wrapper calls async implementation."""
        # Mock scan to return category keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"category:planning"],
        )

        # Mock scard to return category count
        mock_glide_client.scard.return_value = 5

        # Call sync wrapper
        categories = valkey_storage.list_categories(scope_prefix=None)

        # Verify it works
        assert categories == {"planning": 5}


class TestValkeyStorageCount:
    """Tests for count operation."""

    @pytest.mark.asyncio
    async def test_count_all_records_returns_correct_total(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test count returns correct total count across all scopes."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test1", b"scope:/test2"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2"],  # /test1
            [b"record-3", b"record-4", b"record-5"],  # /test2
        ]

        # Count all records
        count = await valkey_storage._acount(scope_prefix=None)

        # Verify total count
        assert count == 5

    @pytest.mark.asyncio
    async def test_count_with_scope_returns_correct_total(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test count with scope filtering returns correct total."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test", b"scope:/test/sub"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2"],  # /test
            [b"record-3"],  # /test/sub
        ]

        # Count records in scope
        count = await valkey_storage._acount(scope_prefix="/test")

        # Verify count includes subscopes
        assert count == 3

    @pytest.mark.asyncio
    async def test_count_empty_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test count returns 0 for empty scope."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/empty"],
        )

        # Mock ZRANGE to return no record IDs
        mock_glide_client.zrange.side_effect = [
            [],  # No records
        ]

        # Count records in empty scope
        count = await valkey_storage._acount(scope_prefix="/empty")

        # Verify count is 0
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_deduplicates_records(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test count deduplicates records that appear in multiple scopes."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test1", b"scope:/test2"],
        )

        # Mock ZRANGE to return overlapping record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2"],  # /test1
            [b"record-2", b"record-3"],  # /test2 (record-2 appears in both)
        ]

        # Count all records
        count = await valkey_storage._acount(scope_prefix=None)

        # Verify count deduplicates (3 unique records, not 4)
        assert count == 3

    def test_count_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync count wrapper calls async implementation."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/test"],
        )

        # Mock ZRANGE to return record IDs
        mock_glide_client.zrange.side_effect = [
            [b"record-1", b"record-2"],
        ]

        # Call sync wrapper
        count = valkey_storage.count(scope_prefix="/test")

        # Verify it works
        assert count == 2


class TestValkeyStorageReset:
    """Tests for reset operation."""

    @pytest.mark.asyncio
    async def test_reset_clears_all_records(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test reset delegates to adelete to clear all records."""
        # Mock adelete to track if it was called
        original_adelete = valkey_storage.adelete
        adelete_called = False
        adelete_args = None

        async def mock_adelete(*args: object, **kwargs: object) -> int:
            nonlocal adelete_called, adelete_args
            adelete_called = True
            adelete_args = kwargs
            return 0

        valkey_storage.adelete = mock_adelete  # type: ignore[method-assign]

        # Reset all records
        await valkey_storage._areset(scope_prefix=None)

        # Verify adelete was called with correct arguments
        assert adelete_called
        assert adelete_args == {"scope_prefix": None}

        # Restore original method
        valkey_storage.adelete = original_adelete  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_reset_with_scope_clears_scope_records(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test reset with scope delegates to adelete with scope_prefix."""
        # Mock adelete to track if it was called
        original_adelete = valkey_storage.adelete
        adelete_called = False
        adelete_args = None

        async def mock_adelete(*args: object, **kwargs: object) -> int:
            nonlocal adelete_called, adelete_args
            adelete_called = True
            adelete_args = kwargs
            return 0

        valkey_storage.adelete = mock_adelete  # type: ignore[method-assign]

        # Reset records in scope
        await valkey_storage._areset(scope_prefix="/test")

        # Verify adelete was called with correct arguments
        assert adelete_called
        assert adelete_args == {"scope_prefix": "/test"}

        # Restore original method
        valkey_storage.adelete = original_adelete  # type: ignore[method-assign]

    def test_reset_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync reset wrapper calls async implementation."""
        # Mock adelete to track if it was called
        original_adelete = valkey_storage.adelete
        adelete_called = False

        async def mock_adelete(*args: object, **kwargs: object) -> int:
            nonlocal adelete_called
            adelete_called = True
            return 0

        valkey_storage.adelete = mock_adelete  # type: ignore[method-assign]

        # Call sync wrapper
        valkey_storage.reset(scope_prefix="/test")

        # Verify adelete was called
        assert adelete_called

        # Restore original method
        valkey_storage.adelete = original_adelete  # type: ignore[method-assign]
