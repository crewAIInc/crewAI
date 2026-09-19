"""Regression tests: LanceDB scope-prefix filters must respect path boundaries.

A filter for scope ``/app`` covers ``/app`` and its descendants (``/app/...``)
but must never match a sibling scope whose name merely shares a string prefix
(``/apple``) or the root scope (``/``).
"""

from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
import pytest


@pytest.fixture
def storage(tmp_path: Path) -> LanceDBStorage:
    s = LanceDBStorage(path=tmp_path / "memory", vector_dim=4, compact_every=0)
    s.save(
        [
            MemoryRecord(
                content="root note", scope="/", categories=["root"], embedding=[0.0] * 4
            ),
            MemoryRecord(
                content="app note", scope="/app", categories=["mine"], embedding=[0.0] * 4
            ),
            MemoryRecord(
                content="app child", scope="/app/sub", categories=["mine"], embedding=[0.0] * 4
            ),
            MemoryRecord(
                content="sibling note", scope="/apple", categories=["mine"], embedding=[0.0] * 4
            ),
        ]
    )
    return s


def test_search_scope_prefix_excludes_sibling_scope(
    storage: LanceDBStorage,
) -> None:
    results = storage.search([0.0] * 4, scope_prefix="/app", limit=10)
    assert {r.scope for r, _ in results} == {"/app", "/app/sub"}


def test_list_records_scope_prefix_excludes_sibling_scope(
    storage: LanceDBStorage,
) -> None:
    records = storage.list_records(scope_prefix="/app")
    assert {r.scope for r in records} == {"/app", "/app/sub"}


def test_scope_info_excludes_sibling_scope(storage: LanceDBStorage) -> None:
    info = storage.get_scope_info("/app")
    assert info.record_count == 2
    assert info.categories == ["mine"]


def test_count_excludes_sibling_scope(storage: LanceDBStorage) -> None:
    assert storage.count("/app") == 2


def test_list_categories_excludes_sibling_scope(storage: LanceDBStorage) -> None:
    assert storage.list_categories("/app") == {"mine": 2}


def test_scoped_category_delete_excludes_sibling_scope(
    storage: LanceDBStorage,
) -> None:
    deleted = storage.delete(scope_prefix="/app", categories=["mine"])
    assert deleted == 2
    remaining = {r.scope for r in storage.list_records()}
    assert remaining == {"/", "/apple"}


def test_scope_prefix_without_leading_slash_is_normalized(
    storage: LanceDBStorage,
) -> None:
    records = storage.list_records(scope_prefix="app")
    assert {r.scope for r in records} == {"/app", "/app/sub"}


def test_root_scope_prefix_matches_everything(storage: LanceDBStorage) -> None:
    assert storage.count("/") == 4
    records = storage.list_records(scope_prefix="/")
    assert len(records) == 4


def test_list_scopes_still_returns_children_only(storage: LanceDBStorage) -> None:
    assert storage.list_scopes("/") == ["/app", "/apple"]
    assert storage.list_scopes("/app") == ["/app/sub"]
