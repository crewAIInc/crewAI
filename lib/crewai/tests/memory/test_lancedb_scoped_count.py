"""Scoped record counts must stay exact above the metadata scan cap.

``LanceDBStorage.count(scope)`` and ``get_scope_info(scope).record_count``
used to count rows returned by a metadata scan capped at 50,000 rows, while
the unscoped count used LanceDB's native ``count_rows()``. The two views
disagreed once a scope held more than 50,000 records. Scoped counts now use
the native filtered ``count_rows()`` with the same scope-prefix predicate as
the scan, so the matched set is identical and only the cap goes away. These
tests run against a real temporary LanceDB table.
"""

from __future__ import annotations

from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord

DIM = 2
SCAN_LIMIT = 50_000


def _record(record_id: str, scope: str) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        embedding=[0.1] * DIM,
    )


def _new_storage(path: Path, records: list[MemoryRecord]) -> LanceDBStorage:
    storage = LanceDBStorage(path=path, vector_dim=DIM, compact_every=0)
    storage.save(records)
    return storage


def test_scoped_counts_stay_exact_above_scan_limit(tmp_path: Path) -> None:
    records = [_record(f"team-{i}", "/team") for i in range(25_001)]
    records += [_record(f"sub-{i}", "/team/sub") for i in range(25_000)]
    storage = _new_storage(tmp_path / "mem", records)

    assert storage.count() == 50_001
    assert storage.count("/team") == 50_001
    assert storage.get_scope_info("/team").record_count == 50_001
    assert storage.count("/team/sub") == 25_000
    assert storage.get_scope_info("/team/sub").record_count == 25_000


def test_scoped_counts_match_scan_semantics(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("a", "/team"),
            _record("b", "/team/sub"),
            _record("c", "/teamwork"),
            _record("d", "/other"),
            _record("e", "/a_b"),
            _record("f", "/axb"),
        ],
    )

    assert storage.count("/team") == 3
    assert storage.count("/team/sub") == 1
    assert storage.count("/a_b") == 1
    assert storage.count("/missing") == 0
    assert storage.count("/") == 6
    assert storage.count("team") == 3
    assert storage.count("/team/") == 3

    info = storage.get_scope_info("/team")
    assert info.record_count == 3
    assert info.child_scopes == ["/team/sub"]
    assert storage.get_scope_info("/").record_count == 6
    assert storage.get_scope_info("/missing").record_count == 0
