"""Scope and record-id values must be matched literally in LanceDB filters.

LanceDB ``where()`` takes a raw SQL expression, so every caller-supplied value
has to be escaped before it is embedded. These tests run against a real
temporary LanceDB table with two tenants and check that crafted scope prefixes
and record ids cannot read, delete, or reset another tenant's records, and that
legitimate scope names (including ``-``, ``_``, ``/``, ``%``, quotes, and
backslashes) keep working.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest


DIM = 4
TENANT_A = {"a1", "a2"}
TENANT_B = {"b1", "b2"}

CRAFTED_SCOPES = [
    pytest.param("/tenant-a' OR scope <> '", id="quote-breakout"),
    pytest.param("/tenant-a' OR scope LIKE '/tenant-b", id="predicate-injection"),
    pytest.param("%", id="bare-percent"),
    pytest.param("_", id="bare-underscore"),
    pytest.param("/tenant-_", id="underscore-wildcard"),
]


def _record(
    record_id: str, scope: str, categories: list[str] | None = None
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        categories=categories or ["note"],
        embedding=[0.1] * DIM,
    )


def _new_storage(path: Path, records: list[MemoryRecord]) -> LanceDBStorage:
    storage = LanceDBStorage(path=path, vector_dim=DIM, compact_every=0)
    storage.save(records)
    return storage


def _all_ids(storage: LanceDBStorage) -> set[str]:
    return {r.id for r in storage.list_records(limit=1000)}


def _search_ids(storage: LanceDBStorage, scope_prefix: str) -> set[str]:
    return {
        r.id
        for r, _ in storage.search([0.1] * DIM, scope_prefix=scope_prefix, limit=100)
    }


@pytest.fixture
def storage(tmp_path: Path) -> LanceDBStorage:
    return _new_storage(
        tmp_path / "mem",
        [
            _record("a1", "/tenant-a/agent_1"),
            _record("a2", "/tenant-a/notes"),
            _record("b1", "/tenant-b/agent_1"),
            _record("b2", "/tenant-b/secrets"),
        ],
    )


def test_control_scope_prefix_stays_in_tenant(storage: LanceDBStorage) -> None:
    assert _search_ids(storage, "/tenant-a") == TENANT_A
    assert {r.id for r in storage.list_records("/tenant-a")} == TENANT_A
    assert storage.count("/tenant-a") == 2


@pytest.mark.parametrize("payload", CRAFTED_SCOPES)
def test_crafted_scope_prefix_does_not_read_other_tenant(
    storage: LanceDBStorage, payload: str
) -> None:
    assert _search_ids(storage, payload) <= TENANT_A
    assert {r.id for r in storage.list_records(payload)} <= TENANT_A
    assert storage.count(payload) <= len(TENANT_A)


@pytest.mark.parametrize("payload", CRAFTED_SCOPES)
@pytest.mark.parametrize(
    "remove",
    [
        pytest.param(lambda s, p: s.delete(scope_prefix=p), id="delete"),
        pytest.param(
            lambda s, p: s.delete(scope_prefix=p, categories=["note"]),
            id="delete-by-category",
        ),
        pytest.param(lambda s, p: s.reset(scope_prefix=p), id="reset"),
    ],
)
def test_crafted_scope_prefix_does_not_remove_other_tenant(
    storage: LanceDBStorage,
    payload: str,
    remove: Callable[[LanceDBStorage, str], object],
) -> None:
    remove(storage, payload)
    assert _all_ids(storage) >= TENANT_B


def test_crafted_record_id_does_not_over_delete(storage: LanceDBStorage) -> None:
    deleted = storage.delete(record_ids=["a1') OR id <> ('"])
    assert deleted == 0
    assert _all_ids(storage) == TENANT_A | TENANT_B


def test_record_ids_with_quotes_are_matched_literally(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("o'brien-1", "/people/o'brien", ["x"]),
            _record("o'brien-2", "/people/o'brien", ["x"]),
            _record("other", "/people/other", ["x"]),
        ],
    )
    deleted_by_id = storage.delete(record_ids=["o'brien-1"])
    assert deleted_by_id == 1
    deleted_by_scope = storage.delete(scope_prefix="/people/o'brien", categories=["x"])
    assert deleted_by_scope == 1
    assert _all_ids(storage) == {"other"}


def test_memory_root_scope_confines_crafted_explicit_scope(
    storage: LanceDBStorage,
) -> None:
    embedder = MagicMock(side_effect=lambda texts: [[0.1] * DIM for _ in texts])
    memory = Memory(
        storage=storage, root_scope="/tenant-a", llm=MagicMock(), embedder=embedder
    )

    for payload in ("x' OR scope <> '", "x' OR scope LIKE '/tenant-b"):
        matches = memory.recall("anything", scope=payload, depth="shallow")
        assert {m.record.id for m in matches} <= TENANT_A
        memory.forget(scope=payload)
        assert _all_ids(storage) == TENANT_A | TENANT_B

    assert {m.record.id for m in memory.recall("anything", depth="shallow")} == TENANT_A


def test_legitimate_scope_characters_still_match(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("senior", "/crew/research-crew/agent/senior_analyst"),
            _record("junior", "/crew/research-crew/agent/junior-analyst"),
            _record("underscore", "/crew/research_crew"),
            _record("lookalike", "/crew/researchXcrew"),
        ],
    )
    assert _search_ids(storage, "/crew/research-crew") == {"senior", "junior"}
    assert _search_ids(storage, "/crew/research-crew/agent/senior_analyst") == {
        "senior"
    }
    assert storage.get_scope_info("/crew/research-crew").record_count == 2
    assert storage.list_scopes("/crew/research-crew") == ["/crew/research-crew/agent"]
    assert storage.list_categories("/crew/research-crew") == {"note": 2}

    deleted = storage.delete(scope_prefix="/crew/research_crew")
    assert deleted == 1
    assert _all_ids(storage) == {"senior", "junior", "lookalike"}

    storage.reset(scope_prefix="/crew/research-crew")
    assert _all_ids(storage) == {"lookalike"}


def test_percent_underscore_and_backslash_in_scope_names_match_literally(
    tmp_path: Path,
) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("pct", "/metrics/100%"),
            _record("pct-child", "/metrics/100%/daily"),
            _record("digits", "/metrics/1000"),
            _record("underscore", "/a_b"),
            _record("lookalike", "/axb"),
            _record("backslash", "/share\\docs"),
        ],
    )
    assert _search_ids(storage, "/metrics/100%") == {"pct", "pct-child"}
    assert {r.id for r in storage.list_records("/metrics/100%")} == {"pct", "pct-child"}
    assert _search_ids(storage, "/a_b") == {"underscore"}
    assert storage.count("/a_b") == 1
    assert _search_ids(storage, "/share\\") == {"backslash"}

    deleted = storage.delete(scope_prefix="/a_b")
    assert deleted == 1
    assert _all_ids(storage) == {"pct", "pct-child", "digits", "lookalike", "backslash"}
