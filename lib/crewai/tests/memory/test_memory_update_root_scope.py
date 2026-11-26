"""``Memory.update()`` must honour ``root_scope`` like every other scope-taking method.

``update()`` looked the record up by bare ID and wrote the ``scope`` argument
through unchanged, so a ``Memory`` built with ``root_scope="/t1"`` could rewrite
-- and move -- a record belonging to ``/t2``. Every other method that takes a
scope (``recall``, ``forget``, ``reset``, ``list_scopes``, ``list_records``,
``info``, ``tree``, ``list_categories``) joins its argument under ``root_scope``
first, and the read paths additionally confine their results to it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest

DIM = 4
EMB = [0.1] * DIM


def _embedder() -> MagicMock:
    m = MagicMock()
    m.side_effect = lambda texts: [EMB for _ in texts]
    return m


def _build(
    tmp_path: Path, root_scope: str | None = "/t1", read_only: bool = False
) -> tuple[Memory, LanceDBStorage]:
    storage = LanceDBStorage(path=tmp_path / "mem", vector_dim=DIM, compact_every=0)
    storage.save(
        [
            MemoryRecord(id="inside", content="mine", scope="/t1/a", embedding=EMB),
            MemoryRecord(id="deep", content="nested", scope="/t1/a/b", embedding=EMB),
            MemoryRecord(id="sibling", content="lookalike", scope="/t1x", embedding=EMB),
            MemoryRecord(id="outside", content="theirs", scope="/t2", embedding=EMB),
        ]
    )
    memory = Memory(
        storage=storage,
        root_scope=root_scope,
        read_only=read_only,
        llm=MagicMock(),
        embedder=_embedder(),
    )
    return memory, storage


def test_update_outside_root_scope_is_rejected(tmp_path: Path) -> None:
    memory, storage = _build(tmp_path)

    with pytest.raises(ValueError, match="Record not found: outside"):
        memory.update("outside", content="tampered")

    assert storage.get_record("outside").content == "theirs"


@pytest.mark.parametrize(
    "record_id",
    [pytest.param("inside", id="direct-child"), pytest.param("deep", id="nested")],
)
def test_update_inside_root_scope_works(tmp_path: Path, record_id: str) -> None:
    memory, storage = _build(tmp_path)

    updated = memory.update(record_id, content="rewritten")

    assert updated.content == "rewritten"
    assert storage.get_record(record_id).content == "rewritten"


def test_sibling_scope_is_not_inside_root_scope(tmp_path: Path) -> None:
    # '/t1x' starts with '/t1' as a string but is a different path component.
    memory, storage = _build(tmp_path)

    with pytest.raises(ValueError, match="Record not found: sibling"):
        memory.update("sibling", importance=0.9)

    assert storage.get_record("sibling").importance != 0.9


def test_new_scope_is_joined_under_root_scope(tmp_path: Path) -> None:
    memory, _ = _build(tmp_path)

    updated = memory.update("inside", scope="/notes")

    assert updated.scope == "/t1/notes"


def test_read_only_update_does_not_reveal_an_outside_record(tmp_path: Path) -> None:
    memory, _ = _build(tmp_path, read_only=True)

    with pytest.raises(ValueError, match="Record not found: outside"):
        memory.update("outside")


def test_missing_record_still_raises(tmp_path: Path) -> None:
    memory, _ = _build(tmp_path)

    with pytest.raises(ValueError, match="Record not found: nope"):
        memory.update("nope", content="x")


def test_without_root_scope_every_record_is_reachable(tmp_path: Path) -> None:
    memory, _ = _build(tmp_path, root_scope=None)

    updated = memory.update("outside", content="rewritten", scope="/elsewhere")

    assert updated.content == "rewritten"
    assert updated.scope == "/elsewhere"


def test_root_scope_of_slash_reaches_every_record(tmp_path: Path) -> None:
    memory, _ = _build(tmp_path, root_scope="/")

    assert memory.update("outside", content="rewritten").content == "rewritten"
