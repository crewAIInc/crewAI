"""Scoped delete must stay inside the scope.

``MemoryRecord``'s default scope is ``"/"``, so root-level memories are the
norm for unscoped ``Memory.remember()``. ``delete(scope_prefix=...)`` used to
OR in ``scope = '/'``, so any scoped forget also removed every root-scope
record — while ``search()``, ``list_records()``, scoped ``reset()`` and the
Qdrant backend's scoped delete all stay inside the prefix and spare ``/``.
"""

from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


def test_delete_scope_prefix_spares_root_records(tmp_path: Path) -> None:
    storage = LanceDBStorage(path=tmp_path / "memory", vector_dim=2, compact_every=0)
    root = MemoryRecord(content="root note", embedding=[1.0, 0.0])  # scope="/"
    scoped = MemoryRecord(content="scoped note", scope="/notes", embedding=[1.0, 0.0])
    other = MemoryRecord(content="other note", scope="/other", embedding=[1.0, 0.0])
    storage.save([root, scoped, other])

    deleted = storage.delete(scope_prefix="/notes")

    assert deleted == 1
    assert storage.get_record(root.id) is not None, "root record must survive"
    assert storage.get_record(scoped.id) is None
    assert storage.get_record(other.id) is not None


def test_delete_without_scope_still_clears_everything(tmp_path: Path) -> None:
    storage = LanceDBStorage(path=tmp_path / "memory", vector_dim=2, compact_every=0)
    root = MemoryRecord(content="root note", embedding=[1.0, 0.0])
    scoped = MemoryRecord(content="scoped note", scope="/notes", embedding=[1.0, 0.0])
    storage.save([root, scoped])

    assert storage.delete() == 2
    assert storage.get_record(root.id) is None
    assert storage.get_record(scoped.id) is None
