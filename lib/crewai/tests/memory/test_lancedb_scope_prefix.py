"""Scope-prefix operations must stay inside their own scope subtree.

``delete``/``search``/``count`` with ``scope_prefix`` must match the scope
itself and its descendants — never sibling scopes that merely share a string
prefix (``/proj`` vs ``/project``), never unrelated root-scope records, and
an ``older_than`` cutoff combined with ``scope_prefix`` must apply to the
whole subtree.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord

DIM = 4


def _record(scope: str, days_old: int = 0) -> MemoryRecord:
    created = datetime.utcnow() - timedelta(days=days_old)
    return MemoryRecord(
        content=f"content at {scope}",
        scope=scope,
        embedding=[0.1] * DIM,
        created_at=created,
        last_accessed=created,
    )


def _scopes(storage: LanceDBStorage) -> list[str]:
    return sorted(r.scope for r in storage.list_records(limit=1000))


def _make_storage(path: Path, records: list[MemoryRecord]) -> LanceDBStorage:
    storage = LanceDBStorage(path=path / "mem", vector_dim=DIM, compact_every=0)
    storage.save(records)
    return storage


def test_delete_scope_prefix_spares_siblings_and_root(tmp_path: Path):
    storage = _make_storage(
        tmp_path,
        [_record("/proj"), _record("/proj/sub"), _record("/project"), _record("/")],
    )

    deleted = storage.delete(scope_prefix="/proj")

    assert deleted == 2
    assert _scopes(storage) == ["/", "/project"]


def test_search_scope_prefix_excludes_siblings(tmp_path: Path):
    storage = _make_storage(
        tmp_path,
        [_record("/proj"), _record("/proj/sub"), _record("/project"), _record("/")],
    )

    hits = storage.search([0.1] * DIM, scope_prefix="/proj", limit=100)

    assert sorted(r.scope for r, _ in hits) == ["/proj", "/proj/sub"]


def test_count_scope_prefix_excludes_siblings(tmp_path: Path):
    storage = _make_storage(
        tmp_path,
        [_record("/proj"), _record("/proj/sub"), _record("/project")],
    )

    assert storage.count("/proj") == 2


def test_delete_scope_prefix_older_than_applies_to_subtree(tmp_path: Path):
    storage = _make_storage(
        tmp_path,
        [_record("/keep/fresh"), _record("/keep/old", days_old=30), _record("/")],
    )

    deleted = storage.delete(
        scope_prefix="/keep",
        older_than=datetime.utcnow() - timedelta(days=1),
    )

    assert deleted == 1
    assert _scopes(storage) == ["/", "/keep/fresh"]


def test_delete_scope_prefix_older_than_can_match_all(tmp_path: Path):
    storage = _make_storage(
        tmp_path, [_record("/keep/a"), _record("/keep/b"), _record("/")]
    )

    deleted = storage.delete(
        scope_prefix="/keep",
        older_than=datetime.utcnow() + timedelta(days=1),
    )

    assert deleted == 2
    assert _scopes(storage) == ["/"]
