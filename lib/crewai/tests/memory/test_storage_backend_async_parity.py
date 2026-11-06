"""Async parity and restart-persistence contract tests for memory backends.

Builds on the shared backend fixtures in ``test_storage_backend_contract``.
Every test runs against each local backend so that:

* ``asave``/``asearch``/``adelete`` observably match ``save``/``search``/
  ``delete`` on equivalent, isolated stores;
* records written to a store survive closing it and reopening the same path;
* dimension errors are raised explicitly and leave the store untouched.

No test exercises locking or concurrency; parity is checked by driving two
separate stores, one through the sync API and one through the async API.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crewai.memory.storage.backend import (
    EmbeddingDimensionMismatchError,
    StorageBackend,
)
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
import pytest

from tests.memory.test_storage_backend_contract import (
    BASE_TIME,
    LANCEDB,
    QDRANT_EDGE,
    VECTOR_DIM,
    _assert_same_record,
    _record,
)


QUERY = [1.0, 0.0, 0.0, 1.0]
WRONG_DIM_EMBEDDING = [1.0, 0.0, 0.0]


@runtime_checkable
class _Closable(Protocol):
    def close(self) -> None: ...


def _open_storage(
    backend: str, path: Path, vector_dim: int | None = VECTOR_DIM
) -> StorageBackend:
    """Open a backend of kind ``backend`` rooted at ``path``."""
    if backend == "lancedb":
        return LanceDBStorage(path=str(path), vector_dim=vector_dim)

    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

    return QdrantEdgeStorage(path=str(path), vector_dim=vector_dim)


def _close_storage(storage: StorageBackend) -> None:
    """Flush and release a backend; a no-op for backends without ``close``."""
    if isinstance(storage, _Closable):
        storage.close()


def _reopen(
    backend: str,
    storage: StorageBackend,
    path: Path,
    vector_dim: int | None = VECTOR_DIM,
) -> StorageBackend:
    """Simulate a process restart: close ``storage`` and reopen ``path``."""
    _close_storage(storage)
    return _open_storage(backend, path, vector_dim=vector_dim)


def _fixture_records() -> list[MemoryRecord]:
    """Four records that differ in scope, category, metadata, and timestamp."""
    return [
        _record(
            0,
            scope="/team/a",
            categories=["alpha"],
            metadata={"env": "prod"},
            importance=0.9,
            source="user-1",
            private=True,
        ),
        _record(1, scope="/team/a/sub", categories=["beta"], metadata={"env": "dev"}),
        _record(
            2, scope="/team/b", categories=["alpha", "beta"], metadata={"env": "prod"}
        ),
        _record(3, scope="/other"),
    ]


def _search_ids(results: list[tuple[MemoryRecord, float]]) -> list[str]:
    return [record.id for record, _ in results]


def _search_scores(results: list[tuple[MemoryRecord, float]]) -> list[float]:
    return [score for _, score in results]


@pytest.fixture(params=[LANCEDB, QDRANT_EDGE])
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def storage(backend: str, tmp_path: Path) -> Iterator[StorageBackend]:
    """Yield an empty backend rooted in a temporary directory."""
    store = _open_storage(backend, tmp_path / "store")
    yield store
    _close_storage(store)


@pytest.fixture
def storage_pair(
    backend: str, tmp_path: Path
) -> Iterator[tuple[StorageBackend, StorageBackend]]:
    """Yield two isolated, empty backends of the same kind.

    The first is driven through the sync API and the second through the async
    API, so a parity test never shares a store (or its locks) between the two.
    """
    sync_store = _open_storage(backend, tmp_path / "sync")
    async_store = _open_storage(backend, tmp_path / "async")
    yield sync_store, async_store
    _close_storage(sync_store)
    _close_storage(async_store)


@pytest.mark.asyncio
class TestAsyncParity:
    async def test_asave_persists_the_same_records_as_save(
        self, storage_pair: tuple[StorageBackend, StorageBackend]
    ) -> None:
        sync_store, async_store = storage_pair
        records = _fixture_records()

        sync_store.save(records)
        await async_store.asave(records)

        assert async_store.count() == sync_store.count() == len(records)
        for record in records:
            _assert_same_record(async_store.get_record(record.id), record)
        assert [r.id for r in async_store.list_records()] == [
            r.id for r in sync_store.list_records()
        ]

    @pytest.mark.parametrize(
        "search_kwargs",
        [
            pytest.param({}, id="no-filter"),
            pytest.param({"scope_prefix": "/team/a"}, id="scope-prefix"),
            pytest.param({"categories": ["alpha"]}, id="categories"),
            pytest.param({"metadata_filter": {"env": "prod"}}, id="metadata"),
            pytest.param({"limit": 2}, id="limit"),
            pytest.param({"min_score": 0.9}, id="min-score"),
        ],
    )
    async def test_asearch_returns_the_same_matches_as_search(
        self,
        storage_pair: tuple[StorageBackend, StorageBackend],
        search_kwargs: dict[str, Any],
    ) -> None:
        sync_store, async_store = storage_pair
        records = _fixture_records()
        sync_store.save(records)
        async_store.save(records)

        sync_results = sync_store.search(QUERY, **search_kwargs)
        async_results = await async_store.asearch(QUERY, **search_kwargs)

        assert 0 < len(sync_results) < len(records) or search_kwargs == {}
        assert _search_ids(async_results) == _search_ids(sync_results)
        assert _search_scores(async_results) == pytest.approx(
            _search_scores(sync_results)
        )
        for (sync_record, _), (async_record, _) in zip(
            sync_results, async_results, strict=True
        ):
            _assert_same_record(async_record, sync_record)

    @pytest.mark.parametrize(
        ("delete_kwargs", "expected_deleted"),
        [
            pytest.param({"record_ids": [0, 2]}, 2, id="record-ids"),
            pytest.param({"scope_prefix": "/team/a"}, 2, id="scope-prefix"),
            pytest.param({"categories": ["alpha"]}, 2, id="categories"),
            pytest.param({"metadata_filter": {"env": "prod"}}, 2, id="metadata"),
            pytest.param(
                {"older_than": BASE_TIME + timedelta(minutes=2)}, 2, id="older-than"
            ),
            pytest.param(
                {"scope_prefix": "/team", "categories": ["beta"]},
                2,
                id="scope-and-categories",
            ),
            pytest.param({}, 4, id="everything"),
        ],
    )
    async def test_adelete_removes_the_same_records_as_delete(
        self,
        storage_pair: tuple[StorageBackend, StorageBackend],
        delete_kwargs: dict[str, Any],
        expected_deleted: int,
    ) -> None:
        sync_store, async_store = storage_pair
        records = _fixture_records()
        sync_store.save(records)
        async_store.save(records)
        if "record_ids" in delete_kwargs:
            delete_kwargs = {
                "record_ids": [records[i].id for i in delete_kwargs["record_ids"]]
            }

        sync_deleted = sync_store.delete(**delete_kwargs)
        async_deleted = await async_store.adelete(**delete_kwargs)

        assert sync_deleted == async_deleted == expected_deleted
        sync_remaining = sorted(r.id for r in sync_store.list_records())
        async_remaining = sorted(r.id for r in async_store.list_records())
        assert async_remaining == sync_remaining
        assert len(sync_remaining) == len(records) - expected_deleted
        assert async_store.count() == sync_store.count() == len(sync_remaining)

    async def test_async_calls_on_an_empty_store_match_sync(
        self, storage_pair: tuple[StorageBackend, StorageBackend]
    ) -> None:
        sync_store, async_store = storage_pair

        assert await async_store.asearch(QUERY) == sync_store.search(QUERY) == []
        assert await async_store.adelete() == sync_store.delete() == 0
        assert await async_store.adelete(record_ids=["missing"]) == 0
        assert async_store.count() == sync_store.count() == 0


class TestRestartPersistence:
    def test_records_survive_reopen(self, backend: str, tmp_path: Path) -> None:
        path = tmp_path / "store"
        storage = _open_storage(backend, path)
        records = _fixture_records()
        storage.save(records)
        listed_before = [r.id for r in storage.list_records()]

        storage = _reopen(backend, storage, path)
        try:
            assert storage.count() == len(records)
            for record in records:
                _assert_same_record(storage.get_record(record.id), record)
            assert [r.id for r in storage.list_records()] == listed_before
        finally:
            _close_storage(storage)

    def test_scope_and_category_views_survive_reopen(
        self, backend: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "store"
        storage = _open_storage(backend, path)
        storage.save(_fixture_records())
        before = {
            "team_count": storage.count(scope_prefix="/team"),
            "sub_count": storage.count(scope_prefix="/team/a"),
            "scopes": sorted(storage.list_scopes("/")),
            "child_scopes": sorted(storage.list_scopes("/team")),
            "categories": storage.list_categories(),
            "team_categories": storage.list_categories(scope_prefix="/team"),
            "team_a_records": [
                r.id for r in storage.list_records(scope_prefix="/team/a")
            ],
            "scope_info": storage.get_scope_info("/team").model_dump(),
        }
        assert before["team_count"] == 3
        assert before["categories"] == {"alpha": 2, "beta": 2}

        storage = _reopen(backend, storage, path)
        try:
            after = {
                "team_count": storage.count(scope_prefix="/team"),
                "sub_count": storage.count(scope_prefix="/team/a"),
                "scopes": sorted(storage.list_scopes("/")),
                "child_scopes": sorted(storage.list_scopes("/team")),
                "categories": storage.list_categories(),
                "team_categories": storage.list_categories(scope_prefix="/team"),
                "team_a_records": [
                    r.id for r in storage.list_records(scope_prefix="/team/a")
                ],
                "scope_info": storage.get_scope_info("/team").model_dump(),
            }
        finally:
            _close_storage(storage)

        assert after == before

    def test_search_results_survive_reopen(self, backend: str, tmp_path: Path) -> None:
        path = tmp_path / "store"
        storage = _open_storage(backend, path)
        storage.save(_fixture_records())
        before = storage.search(QUERY, scope_prefix="/team", categories=["alpha"])
        assert len(before) == 2

        storage = _reopen(backend, storage, path)
        try:
            after = storage.search(QUERY, scope_prefix="/team", categories=["alpha"])
        finally:
            _close_storage(storage)

        assert _search_ids(after) == _search_ids(before)
        assert _search_scores(after) == pytest.approx(_search_scores(before))

    @pytest.mark.asyncio
    async def test_async_saved_records_survive_reopen(
        self, backend: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "store"
        storage = _open_storage(backend, path)
        records = _fixture_records()
        await storage.asave(records)

        storage = _reopen(backend, storage, path)
        try:
            assert storage.count() == len(records)
            for record in records:
                _assert_same_record(storage.get_record(record.id), record)
            found = await storage.asearch(QUERY, scope_prefix="/team/a")
            assert sorted(_search_ids(found)) == sorted([records[0].id, records[1].id])
        finally:
            _close_storage(storage)

    def test_update_and_delete_survive_reopen(
        self, backend: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "store"
        storage = _open_storage(backend, path)
        records = _fixture_records()
        storage.save(records)
        updated = records[0].model_copy(
            update={
                "content": "rewritten",
                "categories": ["revised"],
                "metadata": {"version": 2},
                "scope": "/team/moved",
            }
        )
        storage.update(updated)
        assert storage.delete(record_ids=[records[3].id]) == 1

        storage = _reopen(backend, storage, path)
        try:
            assert storage.count() == len(records) - 1
            _assert_same_record(storage.get_record(updated.id), updated)
            assert storage.get_record(records[3].id) is None
            _assert_same_record(storage.get_record(records[1].id), records[1])
            _assert_same_record(storage.get_record(records[2].id), records[2])
        finally:
            _close_storage(storage)

    def test_reopen_without_vector_dim_infers_it_from_the_store(
        self, backend: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "store"
        storage = _open_storage(backend, path)
        records = _fixture_records()
        storage.save(records)

        storage = _reopen(backend, storage, path, vector_dim=None)
        try:
            assert _search_ids(storage.search(QUERY, limit=1)) == [records[0].id]
            extra = _record(4, scope="/team/a")
            storage.save([extra])
            _assert_same_record(storage.get_record(extra.id), extra)
            with pytest.raises(EmbeddingDimensionMismatchError):
                storage.save([_record(5, embedding=WRONG_DIM_EMBEDDING)])
            assert storage.count() == len(records) + 1
        finally:
            _close_storage(storage)


class TestStorageErrors:
    @pytest.mark.asyncio
    async def test_save_rejects_a_wrong_dimension_vector(
        self, storage: StorageBackend
    ) -> None:
        records = _fixture_records()
        storage.save(records)
        bad = _record(9, embedding=WRONG_DIM_EMBEDDING)

        with pytest.raises(EmbeddingDimensionMismatchError) as sync_error:
            storage.save([bad])
        with pytest.raises(EmbeddingDimensionMismatchError) as async_error:
            await storage.asave([bad])

        for error in (sync_error.value, async_error.value):
            assert error.stored_dim == VECTOR_DIM
            assert error.new_dim == len(WRONG_DIM_EMBEDDING)
            assert isinstance(error, ValueError)
            assert not isinstance(error, RuntimeError)
        assert storage.get_record(bad.id) is None
        assert storage.count() == len(records)

    @pytest.mark.asyncio
    async def test_search_rejects_a_wrong_dimension_query(
        self, storage: StorageBackend
    ) -> None:
        storage.save(_fixture_records())

        with pytest.raises(EmbeddingDimensionMismatchError) as sync_error:
            storage.search(WRONG_DIM_EMBEDDING)
        with pytest.raises(EmbeddingDimensionMismatchError) as async_error:
            await storage.asearch(WRONG_DIM_EMBEDDING)

        assert sync_error.value.stored_dim == async_error.value.stored_dim == VECTOR_DIM
        assert sync_error.value.new_dim == async_error.value.new_dim == 3

    def test_mixed_dimension_batch_writes_nothing(
        self, storage: StorageBackend
    ) -> None:
        good = _record(0)
        bad = _record(1, embedding=WRONG_DIM_EMBEDDING)

        with pytest.raises(EmbeddingDimensionMismatchError) as error:
            storage.save([good, bad])

        assert (error.value.stored_dim, error.value.new_dim) == (VECTOR_DIM, 3)
        assert storage.count() == 0
        assert storage.get_record(good.id) is None

    def test_update_with_a_wrong_dimension_vector_preserves_the_record(
        self, storage: StorageBackend
    ) -> None:
        original = _record(0, content="original")
        storage.save([original])
        replacement = original.model_copy(
            update={"content": "replacement", "embedding": WRONG_DIM_EMBEDDING}
        )

        with pytest.raises(EmbeddingDimensionMismatchError):
            storage.update(replacement)

        _assert_same_record(storage.get_record(original.id), original)
        assert storage.count() == 1
