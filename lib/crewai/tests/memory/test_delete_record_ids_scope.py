"""Deleting by record ID must stay inside the requested scope.

``delete(record_ids=...)`` took a fast path in both backends that dropped
``scope_prefix`` entirely, so a caller confined to one subtree deleted records
anywhere in the store. ``Memory.forget(record_ids=...)`` is the user-facing
case: its docstring promises "If None and root_scope is set, deletes only
within root_scope", and it does pass ``scope_prefix=root_scope`` down -- the
storage layer just ignored it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest

if TYPE_CHECKING:
    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

DIM = 4
EMB = [0.1] * DIM


def _record(record_id: str, scope: str) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        categories=["note"],
        embedding=EMB,
    )


def _lance(tmp_path: Path) -> LanceDBStorage:
    storage = LanceDBStorage(path=tmp_path / "mem", vector_dim=DIM, compact_every=0)
    storage.save([_record("mine", "/t1/a"), _record("theirs", "/t2")])
    return storage


def _ids(storage: LanceDBStorage | QdrantEdgeStorage) -> set[str]:
    return {r.id for r in storage.list_records(limit=100)}


def test_lance_ids_outside_scope_are_not_deleted(tmp_path: Path) -> None:
    storage = _lance(tmp_path)

    deleted = storage.delete(record_ids=["theirs"], scope_prefix="/t1")

    assert deleted == 0
    assert _ids(storage) == {"mine", "theirs"}


def test_lance_ids_inside_scope_are_deleted(tmp_path: Path) -> None:
    storage = _lance(tmp_path)

    deleted = storage.delete(record_ids=["mine"], scope_prefix="/t1")

    assert deleted == 1
    assert _ids(storage) == {"theirs"}


def test_lance_mixed_ids_delete_only_the_scoped_one(tmp_path: Path) -> None:
    storage = _lance(tmp_path)

    deleted = storage.delete(record_ids=["mine", "theirs"], scope_prefix="/t1")

    assert deleted == 1
    assert _ids(storage) == {"theirs"}


def test_lance_ids_without_scope_still_delete_everywhere(tmp_path: Path) -> None:
    storage = _lance(tmp_path)

    assert storage.delete(record_ids=["theirs"]) == 1
    assert _ids(storage) == {"mine"}

    assert storage.delete(record_ids=["mine"], scope_prefix="/") == 1
    assert _ids(storage) == set()


def test_memory_forget_by_id_stays_inside_root_scope(tmp_path: Path) -> None:
    storage = _lance(tmp_path)
    memory = Memory(
        storage=storage,
        root_scope="/t1",
        llm=MagicMock(),
        embedder=MagicMock(side_effect=lambda texts: [EMB for _ in texts]),
    )

    deleted = memory.forget(record_ids=["theirs"])

    assert deleted == 0
    assert _ids(storage) == {"mine", "theirs"}

    assert memory.forget(record_ids=["mine"]) == 1
    assert _ids(storage) == {"theirs"}


needs_qdrant = pytest.mark.skipif(
    importlib.util.find_spec("qdrant_edge") is None,
    reason="qdrant-edge-py not installed",
)


def _qdrant(tmp_path: Path) -> QdrantEdgeStorage:
    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

    storage = QdrantEdgeStorage(path=str(tmp_path / "edge"), vector_dim=DIM)
    storage.save([_record("mine", "/t1/a"), _record("theirs", "/t2")])
    return storage


@needs_qdrant
def test_qdrant_ids_outside_scope_are_not_deleted(tmp_path: Path) -> None:
    storage = _qdrant(tmp_path)

    deleted = storage.delete(record_ids=["theirs"], scope_prefix="/t1")

    assert deleted == 0
    assert _ids(storage) == {"mine", "theirs"}


@needs_qdrant
def test_qdrant_ids_inside_scope_are_deleted(tmp_path: Path) -> None:
    storage = _qdrant(tmp_path)

    deleted = storage.delete(record_ids=["mine"], scope_prefix="/t1")

    assert deleted == 1
    assert _ids(storage) == {"theirs"}


@needs_qdrant
def test_qdrant_ids_without_scope_still_delete_everywhere(tmp_path: Path) -> None:
    storage = _qdrant(tmp_path)

    assert storage.delete(record_ids=["theirs"]) == 1
    assert _ids(storage) == {"mine"}
