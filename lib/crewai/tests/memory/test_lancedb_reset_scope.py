"""``reset(scope_prefix=...)`` must clear a scope and its subtree, nothing else.

The predicate was a string range starting at the bare prefix
(``scope >= '/proj' AND scope < '/proj/<U+FFFF>'``), which also covers every
sibling scope whose next character sorts below ``'/'`` (ASCII 47) -- ``'-'`` is
45, ``'.'`` is 46, a space is 32. So ``reset('/proj')`` deleted ``/proj-x``,
``/proj.big`` and ``/proj x`` as well. The qdrant backend delegates to
``delete()`` with component-wise ``scope_ancestors`` matching and never had this
behaviour, so the two backends disagreed about what a reset covers.
"""

from __future__ import annotations

from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord

DIM = 4


def _record(record_id: str, scope: str) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        categories=["note"],
        embedding=[0.1] * DIM,
    )


def _new_storage(path: Path, records: list[MemoryRecord]) -> LanceDBStorage:
    storage = LanceDBStorage(path=path, vector_dim=DIM, compact_every=0)
    storage.save(records)
    return storage


def _ids(storage: LanceDBStorage) -> set[str]:
    return {r.id for r in storage.list_records(limit=1000)}


def test_reset_leaves_sibling_scopes_alone(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("self", "/proj"),
            _record("child", "/proj/team"),
            _record("dash-sib", "/proj-x"),
            _record("dot-sib", "/proj.big"),
            _record("space-sib", "/proj x"),
            _record("other", "/other"),
        ],
    )

    storage.reset(scope_prefix="/proj")

    assert _ids(storage) == {"dash-sib", "dot-sib", "space-sib", "other"}


def test_reset_clears_the_scope_and_its_whole_subtree(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("self", "/proj"),
            _record("child", "/proj/team"),
            _record("grandchild", "/proj/team/agent"),
            _record("other", "/other"),
        ],
    )

    storage.reset(scope_prefix="/proj")

    assert _ids(storage) == {"other"}


def test_reset_of_a_deep_scope_keeps_its_parent(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("parent", "/proj"),
            _record("target", "/proj/team"),
            _record("nephew", "/proj/other"),
        ],
    )

    storage.reset(scope_prefix="/proj/team")

    assert _ids(storage) == {"parent", "nephew"}


def test_reset_without_scope_clears_everything(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem", [_record("a", "/proj"), _record("b", "/other")]
    )

    storage.reset()

    assert _ids(storage) == set()


def test_reset_matches_like_metacharacters_literally(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("pct", "/metrics/100%"),
            _record("pct-child", "/metrics/100%/daily"),
            _record("digits", "/metrics/1000"),
            _record("underscore", "/a_b"),
            _record("lookalike", "/axb"),
        ],
    )

    storage.reset(scope_prefix="/metrics/100%")
    assert _ids(storage) == {"digits", "underscore", "lookalike"}

    storage.reset(scope_prefix="/a_b")
    assert _ids(storage) == {"digits", "lookalike"}


def test_reset_matches_a_scope_containing_a_quote_literally(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("quoted", "/people/o'brien"),
            _record("quoted-child", "/people/o'brien/notes"),
            _record("other", "/people/other"),
        ],
    )

    storage.reset(scope_prefix="/people/o'brien")

    assert _ids(storage) == {"other"}
