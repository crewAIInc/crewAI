"""Shared StorageBackend contract tests for record lifecycle and pagination.

Every test runs against each memory storage backend so behavioral
differences in save, get_record, update, list_records, and count surface
as failures instead of backend-specific surprises.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from typing import Any
from uuid import uuid4

from crewai.memory.storage.backend import StorageBackend
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
import pytest


VECTOR_DIM = 4
BASE_TIME = datetime(2026, 1, 1, 12, 0, 0)

LANCEDB = pytest.param("lancedb", id="lancedb")
QDRANT_EDGE = pytest.param(
    "qdrant_edge",
    id="qdrant_edge",
    marks=pytest.mark.skipif(
        importlib.util.find_spec("qdrant_edge") is None,
        reason="qdrant-edge-py not installed",
    ),
)
LANCEDB_LIST_ORDER_BUG = pytest.param(
    "lancedb",
    id="lancedb",
    marks=pytest.mark.xfail(
        strict=True,
        reason="LanceDB truncates the scan before sorting (#7394); fixed by #7418",
    ),
)


@pytest.fixture(params=[LANCEDB, QDRANT_EDGE])
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StorageBackend]:
    """Yield an empty storage backend rooted in a temporary directory."""
    if request.param == "lancedb":
        yield LanceDBStorage(path=str(tmp_path / "lancedb"), vector_dim=VECTOR_DIM)
        return

    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

    qdrant = QdrantEdgeStorage(path=str(tmp_path / "qdrant"), vector_dim=VECTOR_DIM)
    yield qdrant
    qdrant.close()


def _record(index: int, **overrides: Any) -> MemoryRecord:
    """Build a record with a fixed vector and a timestamp ``index`` minutes after BASE_TIME."""
    fields: dict[str, Any] = {
        "content": f"memory {index}",
        "scope": "/",
        "embedding": [float(index + 1), 0.0, 0.0, 1.0],
        "created_at": BASE_TIME + timedelta(minutes=index),
        "last_accessed": BASE_TIME + timedelta(minutes=index),
    }
    fields.update(overrides)
    return MemoryRecord(**fields)


def _save_shuffled(storage: StorageBackend, count: int) -> list[MemoryRecord]:
    """Save ``count`` records out of chronological order and return them newest first."""
    records = [_record(i) for i in range(count)]
    shuffled = records[1::2] + records[::2]
    storage.save(shuffled)
    return list(reversed(records))


def _assert_same_record(actual: MemoryRecord | None, expected: MemoryRecord) -> None:
    assert actual is not None
    assert actual.id == expected.id
    assert actual.content == expected.content
    assert actual.scope == expected.scope
    assert actual.categories == expected.categories
    assert actual.metadata == expected.metadata
    assert actual.importance == pytest.approx(expected.importance)
    assert actual.created_at == expected.created_at
    assert actual.last_accessed == expected.last_accessed
    assert actual.source == expected.source
    assert actual.private == expected.private
    # Embeddings are not compared: Qdrant Edge stores them unit-normalized for
    # cosine distance, and the contract does not promise exact vectors back.


class TestEmptyStore:
    def test_count_is_zero(self, storage: StorageBackend) -> None:
        assert storage.count() == 0
        assert storage.count(scope_prefix="/a") == 0

    def test_list_records_is_empty(self, storage: StorageBackend) -> None:
        assert storage.list_records() == []

    def test_get_record_returns_none(self, storage: StorageBackend) -> None:
        assert storage.get_record(str(uuid4())) is None


class TestSaveAndGet:
    def test_round_trips_all_fields(self, storage: StorageBackend) -> None:
        record = _record(
            0,
            scope="/team/alpha",
            categories=["decision", "finance"],
            metadata={"ticket": "ABC-1", "priority": 2},
            importance=0.9,
            source="user-42",
            private=True,
        )
        storage.save([record])

        found = storage.get_record(record.id)

        _assert_same_record(found, record)

    def test_batch_save_keeps_records_distinct(self, storage: StorageBackend) -> None:
        records = [_record(i) for i in range(3)]
        storage.save(records)

        assert storage.count() == 3
        for record in records:
            _assert_same_record(storage.get_record(record.id), record)

    def test_missing_id_returns_none_on_populated_store(
        self, storage: StorageBackend
    ) -> None:
        storage.save([_record(0)])

        assert storage.get_record(str(uuid4())) is None


class TestUpdate:
    def test_replaces_only_the_target_record(self, storage: StorageBackend) -> None:
        records = [_record(i) for i in range(3)]
        storage.save(records)
        target = records[1]
        updated = target.model_copy(
            update={
                "content": "rewritten",
                "categories": ["revised"],
                "metadata": {"version": 2},
                "importance": 0.1,
                "embedding": [0.0, 0.0, 1.0, 0.0],
            }
        )

        storage.update(updated)

        _assert_same_record(storage.get_record(target.id), updated)
        _assert_same_record(storage.get_record(records[0].id), records[0])
        _assert_same_record(storage.get_record(records[2].id), records[2])
        assert storage.count() == 3
        assert sorted(r.id for r in storage.list_records()) == sorted(
            r.id for r in records
        )


class TestCount:
    def test_counts_scope_and_subscopes(self, storage: StorageBackend) -> None:
        storage.save(
            [
                _record(0, scope="/a"),
                _record(1, scope="/a/b"),
                _record(2, scope="/c"),
            ]
        )

        assert storage.count() == 3
        assert storage.count(scope_prefix="/") == 3
        assert storage.count(scope_prefix="/a") == 2
        assert storage.count(scope_prefix="/a/b") == 1
        assert storage.count(scope_prefix="/c") == 1
        assert storage.count(scope_prefix="/missing") == 0


class TestListRecords:
    def test_returns_all_records_newest_first(self, storage: StorageBackend) -> None:
        newest_first = _save_shuffled(storage, 5)

        listed = storage.list_records()

        assert [r.id for r in listed] == [r.id for r in newest_first]

    @pytest.mark.parametrize(
        "storage", [LANCEDB_LIST_ORDER_BUG, QDRANT_EDGE], indirect=True
    )
    @pytest.mark.parametrize(
        ("limit", "offset", "expected_indexes"),
        [
            pytest.param(2, 0, [0, 1], id="first-page"),
            pytest.param(2, 2, [2, 3], id="middle-page"),
        ],
    )
    def test_pages_newest_first(
        self,
        storage: StorageBackend,
        limit: int,
        offset: int,
        expected_indexes: list[int],
    ) -> None:
        newest_first = _save_shuffled(storage, 5)

        page = storage.list_records(limit=limit, offset=offset)

        assert [r.id for r in page] == [newest_first[i].id for i in expected_indexes]

    @pytest.mark.parametrize(
        ("limit", "offset", "expected_indexes"),
        [
            pytest.param(2, 4, [4], id="partial-last-page"),
            pytest.param(10, 0, [0, 1, 2, 3, 4], id="limit-exceeds-total"),
            pytest.param(2, 5, [], id="offset-at-end"),
            pytest.param(2, 50, [], id="offset-past-end"),
        ],
    )
    def test_pagination_edges(
        self,
        storage: StorageBackend,
        limit: int,
        offset: int,
        expected_indexes: list[int],
    ) -> None:
        newest_first = _save_shuffled(storage, 5)

        page = storage.list_records(limit=limit, offset=offset)

        assert [r.id for r in page] == [newest_first[i].id for i in expected_indexes]

    def test_scope_prefix_limits_listing(self, storage: StorageBackend) -> None:
        in_scope = [_record(0, scope="/a"), _record(2, scope="/a/b")]
        storage.save([*in_scope, _record(1, scope="/c")])

        listed = storage.list_records(scope_prefix="/a")

        assert [r.id for r in listed] == [in_scope[1].id, in_scope[0].id]
