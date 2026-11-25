"""``LanceDBStorage.delete`` must apply every filter it is given.

Deleting by ``record_ids`` used to run ``id IN (...)`` on its own, ignoring
``scope_prefix`` and ``older_than``. That let a scoped memory delete records
belonging to another scope, and let a just-created record be deleted despite an
``older_than`` cutoff. These tests run against a real temporary LanceDB table.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord

DIM = 2
NOW = datetime(2026, 1, 1, 12, 0, 0)
CUTOFF = NOW - timedelta(days=30)


def _record(
    record_id: str,
    scope: str,
    created_at: datetime = NOW,
    categories: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        created_at=created_at,
        categories=categories or [],
        metadata=metadata or {},
        embedding=[0.1] * DIM,
    )


def _new_storage(path: Path, records: list[MemoryRecord]) -> LanceDBStorage:
    storage = LanceDBStorage(path=path, vector_dim=DIM, compact_every=0)
    storage.save(records)
    return storage


def _exists(storage: LanceDBStorage, record_id: str) -> bool:
    return storage.get_record(record_id) is not None


def test_record_ids_only_deletes_across_scopes(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path, [_record("a", "/tenant1"), _record("b", "/tenant2")]
    )

    assert storage.delete(record_ids=["a", "b"]) == 2
    assert not _exists(storage, "a")
    assert not _exists(storage, "b")


def test_record_ids_do_not_delete_records_outside_scope(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path, [_record("rec1", "/tenant1"), _record("rec2", "/tenant2")]
    )

    assert storage.delete(scope_prefix="/tenant1", record_ids=["rec2"]) == 0
    assert _exists(storage, "rec2")


def test_record_ids_delete_records_inside_scope(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path,
        [
            _record("rec1", "/tenant1/sub"),
            _record("rec2", "/tenant1"),
            _record("rec3", "/tenant2"),
        ],
    )

    assert storage.delete(scope_prefix="/tenant1", record_ids=["rec1", "rec3"]) == 1
    assert not _exists(storage, "rec1")
    assert _exists(storage, "rec2")
    assert _exists(storage, "rec3")


def test_record_ids_respect_older_than(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path,
        [
            _record("fresh", "/t", created_at=NOW),
            _record("stale", "/t", created_at=NOW - timedelta(days=90)),
        ],
    )

    assert storage.delete(record_ids=["fresh", "stale"], older_than=CUTOFF) == 1
    assert _exists(storage, "fresh")
    assert not _exists(storage, "stale")


def test_scope_and_older_than_do_not_delete_other_records(tmp_path: Path) -> None:
    old = NOW - timedelta(days=90)
    storage = _new_storage(
        tmp_path,
        [
            _record("old-in-scope", "/tenant1", created_at=old),
            _record("new-in-scope", "/tenant1", created_at=NOW),
            _record("old-other-scope", "/tenant2", created_at=old),
        ],
    )

    assert storage.delete(scope_prefix="/tenant1", older_than=CUTOFF) == 1
    assert not _exists(storage, "old-in-scope")
    assert _exists(storage, "new-in-scope")
    assert _exists(storage, "old-other-scope")


def test_delete_without_filters_removes_everything(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path, [_record("a", "/tenant1"), _record("b", "/tenant2")]
    )

    assert storage.delete() == 2


def test_record_ids_with_categories_only_delete_listed_ids(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path,
        [
            _record("rec1", "/tenant1", categories=["catA"]),
            _record("rec2", "/tenant2", categories=["catA"]),
            _record("rec3", "/tenant1", categories=["catA"]),
            _record("rec4", "/tenant1", categories=["catB"]),
        ],
    )

    deleted = storage.delete(
        scope_prefix="/tenant1", record_ids=["rec1", "rec4"], categories=["catA"]
    )

    assert deleted == 1
    assert not _exists(storage, "rec1")
    assert _exists(storage, "rec2")
    assert _exists(storage, "rec3")
    assert _exists(storage, "rec4")


def test_record_ids_with_metadata_filter_only_delete_listed_ids(
    tmp_path: Path,
) -> None:
    storage = _new_storage(
        tmp_path,
        [
            _record("a", "/t", metadata={"env": "prod"}),
            _record("b", "/t", metadata={"env": "prod"}),
            _record("c", "/t", metadata={"env": "dev"}),
        ],
    )

    assert storage.delete(record_ids=["a", "c"], metadata_filter={"env": "prod"}) == 1
    assert not _exists(storage, "a")
    assert _exists(storage, "b")
    assert _exists(storage, "c")


def test_categories_without_record_ids_still_delete_all_matches(
    tmp_path: Path,
) -> None:
    storage = _new_storage(
        tmp_path,
        [
            _record("a", "/t", categories=["catA"]),
            _record("b", "/t", categories=["catA"]),
            _record("c", "/t", categories=["catB"]),
        ],
    )

    assert storage.delete(categories=["catA"]) == 2
    assert _exists(storage, "c")


def test_filtered_delete_does_not_load_embeddings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = _new_storage(
        tmp_path,
        [
            _record("a", "/t", categories=["catA"]),
            _record("b", "/t", categories=["catB"]),
        ],
    )
    scanned: list[dict[str, Any]] = []
    original_scan_rows = storage._scan_rows

    def spy(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        rows = original_scan_rows(*args, **kwargs)
        scanned.extend(rows)
        return rows

    monkeypatch.setattr(storage, "_scan_rows", spy)

    assert storage.delete(categories=["catA"]) == 1

    assert scanned
    assert all("vector" not in row for row in scanned)
    assert not _exists(storage, "a")
    assert _exists(storage, "b")


def test_record_ids_are_found_beyond_the_scan_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = _new_storage(
        tmp_path,
        [_record(f"rec{i}", "/t", categories=["catA"]) for i in range(5)],
    )
    original_scan_rows = storage._scan_rows

    def capped_scan(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return original_scan_rows(*args, **{**kwargs, "limit": 2})

    monkeypatch.setattr(storage, "_scan_rows", capped_scan)

    assert storage.delete(record_ids=["rec4"], categories=["catA"]) == 1
    assert not _exists(storage, "rec4")


def test_older_than_with_categories_only_deletes_old_records(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path,
        [
            _record("old", "/t", NOW - timedelta(days=90), categories=["catA"]),
            _record("new", "/t", NOW, categories=["catA"]),
        ],
    )

    assert storage.delete(categories=["catA"], older_than=CUTOFF) == 1
    assert not _exists(storage, "old")
    assert _exists(storage, "new")
