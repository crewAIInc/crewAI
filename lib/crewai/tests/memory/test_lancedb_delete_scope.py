"""LanceDBStorage.delete() scope filtering must not leak unrelated records.

The scope-prefix delete condition was previously built as
``scope LIKE '{prefix}%' OR scope = '/'`` and joined with an ``older_than``
condition via ``" AND ".join(conditions)``. Because SQL ``AND`` binds tighter
than ``OR``, this parsed as
``scope LIKE 'X%' OR (scope = '/' AND created_at < Y)`` -- every record under
the target prefix was deleted unconditionally (the age filter never applied
to it), and every unrelated root-scope (``/``) record was deleted regardless
of prefix.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def lancedb_path(tmp_path: Path) -> Path:
    return tmp_path / "mem"


def _record(scope: str, content: str, created_at: datetime | None = None) -> MemoryRecord:
    kwargs: dict[str, object] = {
        "content": content,
        "scope": scope,
        "embedding": [0.1, 0.2, 0.3, 0.4],
    }
    if created_at is not None:
        kwargs["created_at"] = created_at
    return MemoryRecord(**kwargs)  # type: ignore[arg-type]


def test_delete_by_scope_prefix_does_not_delete_unrelated_root_scope_record(
    lancedb_path: Path,
) -> None:
    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    storage.save([_record("/", "root memory"), _record("/project", "project memory")])

    deleted = storage.delete(scope_prefix="/project")

    assert deleted == 1
    assert storage._table.count_rows() == 1


def test_delete_by_scope_prefix_and_older_than_only_deletes_old_records(
    lancedb_path: Path,
) -> None:
    storage = LanceDBStorage(path=str(lancedb_path), vector_dim=4)
    old_record = _record(
        "/project", "old memory", created_at=datetime.utcnow() - timedelta(days=100)
    )
    new_record = _record("/project", "new memory", created_at=datetime.utcnow())
    storage.save([old_record, new_record])

    cutoff = datetime.utcnow() - timedelta(days=1)
    deleted = storage.delete(scope_prefix="/project", older_than=cutoff)

    assert deleted == 1
    assert storage._table.count_rows() == 1
