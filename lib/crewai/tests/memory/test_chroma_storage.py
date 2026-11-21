"""Tests for the Chroma storage backend."""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from pathlib import Path
import random
from typing import Any
from unittest.mock import MagicMock
import uuid

import pytest

chromadb = pytest.importorskip("chromadb")

from crewai.memory.storage import chroma_storage
from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.storage.chroma_storage import ChromaStorage
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


NOW = datetime(2026, 11, 4, 12, 0, 0, 123456)


@pytest.fixture
def client(tmp_path: Path) -> Any:
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


def _storage(client: Any, vector_dim: int | None = 4) -> ChromaStorage:
    return ChromaStorage(
        client=client, collection_name=f"m_{uuid.uuid4().hex}", vector_dim=vector_dim
    )


def _rec(
    record_id: str,
    scope: str = "/",
    categories: list[str] | None = None,
    embedding: list[float] | None = None,
    **kwargs: Any,
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        categories=categories or [],
        embedding=embedding if embedding is not None else [0.1, 0.2, 0.3, 0.4],
        **kwargs,
    )


SCOPES = {
    "own": "/crew/research",
    "own-child": "/crew/research/agent",
    "dash": "/crew/research-2",
    "longer": "/crew/researcher",
    "root": "/",
}


def _scoped(client: Any) -> ChromaStorage:
    storage = _storage(client)
    storage.save([_rec(k, scope=v, categories=["note"], created_at=NOW) for k, v in SCOPES.items()])
    return storage


def _ids(records: list[MemoryRecord]) -> set[str]:
    return {r.id for r in records}


def test_save_search_get_update(client: Any) -> None:
    storage = _storage(client)
    storage.save([_rec("a", scope="/foo", categories=["cat1"], embedding=[1.0, 0.0, 0.0, 0.0])])
    [(record, score)] = storage.search([0.0, 1.0, 0.0, 0.0], scope_prefix="/foo", limit=5)
    assert record.content == "content of a"
    assert record.categories == ["cat1"]
    assert record.embedding == [1.0, 0.0, 0.0, 0.0]
    # The squared L2 distance is 2: the score is 1 / (1 + 2), as in LanceDBStorage.
    assert score == pytest.approx(1 / 3)
    storage.update(record.model_copy(update={"content": "updated", "categories": []}))
    assert storage.get_record("a").content == "updated"
    assert storage.search([1.0, 0.0, 0.0, 0.0], categories=["cat1"]) == []
    assert storage.count() == 1
    assert storage.get_record("missing") is None


def test_scopes_are_matched_by_path(client: Any) -> None:
    storage = _scoped(client)
    for spelling in ("/crew/research", "/crew/research/", "crew/research"):
        found = storage.search([0.1, 0.2, 0.3, 0.4], scope_prefix=spelling, limit=10)
        assert {r.id for r, _ in found} == {"own", "own-child"}
    assert _ids(storage.list_records("/crew/research")) == {"own", "own-child"}
    assert storage.count("/crew/research") == 2
    assert storage.count("/crew/res") == 0
    assert storage.get_scope_info("/crew/research").child_scopes == ["/crew/research/agent"]
    assert storage.list_categories("/crew/research") == {"note": 2}
    assert storage.list_scopes("/crew") == [
        "/crew/research",
        "/crew/research-2",
        "/crew/researcher",
    ]
    assert storage.count("/") == 5


def test_delete_of_a_scope_keeps_the_root_and_the_other_scopes(client: Any) -> None:
    storage = _scoped(client)
    assert storage.delete(scope_prefix="/crew/research") == 2
    assert _ids(storage.list_records()) == {"dash", "longer", "root"}


def test_delete_of_a_scope_older_than_deletes_only_older_records(client: Any) -> None:
    storage = _scoped(client)
    storage.save([_rec("old", scope="/crew/research", created_at=NOW - timedelta(days=30))])
    assert storage.delete(scope_prefix="/crew/research", older_than=NOW - timedelta(days=1)) == 1
    assert _ids(storage.list_records()) == set(SCOPES)


def test_reset_of_a_scope_and_of_everything(client: Any) -> None:
    storage = _scoped(client)
    storage.reset(scope_prefix="/crew/research")
    assert _ids(storage.list_records()) == {"dash", "longer", "root"}
    storage.reset()
    assert storage.count() == 0
    # After a full reset the store takes another dimension, as LanceDBStorage does.
    storage.save([_rec("wide", embedding=[0.5] * 6)])
    assert len(storage.get_record("wide").embedding) == 6


def test_a_search_by_category_finds_a_far_record(client: Any) -> None:
    storage = _storage(client, vector_dim=2)
    storage.save(
        [
            _rec(
                f"r{i}",
                scope="/crew/a",
                categories=["rare"] if i == 39 else ["common", "line"],
                embedding=[1.0, i / 10],
            )
            for i in range(40)
        ]
        + [_rec("other", scope="/crew/b", categories=["rare"], embedding=[1.0, 0.0])]
    )
    for limit in (1, 5, 13):
        found = storage.search([1.0, 0.0], scope_prefix="/crew/a", categories=["rare"], limit=limit)
        assert [r.id for r, _ in found] == ["r39"]
    found = storage.search([1.0, 0.0], scope_prefix="/crew/a", categories=["rare", "line"], limit=2)
    assert [r.id for r, _ in found] == ["r0", "r1"]


def test_a_categories_list_written_by_another_client_gives_no_categories(client: Any) -> None:
    storage = _storage(client, vector_dim=2)
    storage.save(
        [
            _rec("plain", embedding=[1.0, 0.0]),
            _rec("rare", categories=["rare"], embedding=[0.0, 1.0]),
        ]
    )
    client.get_collection(storage._collection_name, embedding_function=None).update(
        ids=["plain"], metadatas=[{"categories": ["rare"]}]
    )
    assert storage.get_record("plain").categories == []
    assert [r.id for r, _ in storage.search([1.0, 0.0], categories=["rare"], limit=1)] == ["rare"]
    assert storage.list_categories() == {"rare": 1}


def test_search_min_score(client: Any) -> None:
    storage = _storage(client, vector_dim=2)
    storage.save([_rec("near", embedding=[1.0, 0.0]), _rec("far", embedding=[-1.0, 0.0])])
    # Squared distances 0 and 4: scores 1 and 0.2.
    assert [r.id for r, _ in storage.search([1.0, 0.0], limit=5)] == ["near", "far"]
    assert [r.id for r, _ in storage.search([1.0, 0.0], limit=5, min_score=0.5)] == ["near"]


def test_delete_by_category_older_than(client: Any) -> None:
    storage = _storage(client)
    storage.save(
        [
            _rec("old", categories=["c"], created_at=NOW - timedelta(days=30)),
            _rec("new", categories=["c"], created_at=NOW),
            _rec("other", categories=["d"], created_at=NOW - timedelta(days=30)),
        ]
    )
    assert storage.delete(categories=["c"], older_than=NOW - timedelta(days=1)) == 1
    assert _ids(storage.list_records()) == {"new", "other"}


def test_touch_records_updates_last_accessed(client: Any) -> None:
    storage = _storage(client)
    storage.save([_rec("a", last_accessed=NOW), _rec("b", last_accessed=NOW)])
    storage.touch_records(["a", "missing"])
    assert storage.get_record("a").last_accessed > NOW
    assert storage.get_record("b").last_accessed == NOW


def test_list_records_newest_first_with_pagination(client: Any) -> None:
    storage = _storage(client)
    storage.save([_rec(f"r{i}", created_at=NOW + timedelta(minutes=i)) for i in range(5)])
    assert [r.id for r in storage.list_records(limit=2)] == ["r4", "r3"]
    assert [r.id for r in storage.list_records(limit=2, offset=2)] == ["r2", "r1"]
    assert storage.list_records(limit=0) == []


@pytest.mark.parametrize("embedding", [None, []], ids=["none", "empty"])
def test_a_record_without_embedding_is_kept_as_lancedb_keeps_it(
    client: Any, tmp_path: Path, embedding: list[float] | None
) -> None:
    reference = LanceDBStorage(path=tmp_path / "lancedb", vector_dim=3, compact_every=0)
    ours = _storage(client, vector_dim=3)
    records = [
        _rec("with", embedding=[1.0, 2.0, 2.0]),
        MemoryRecord(id="without", content="without", embedding=embedding),
    ]
    for storage in (reference, ours):
        storage.save(records)
    assert ours.get_record("without").embedding == reference.get_record("without").embedding
    query = [0.0, 3.0, 4.0]
    assert [(r.id, round(s, 6)) for r, s in ours.search(query, limit=5)] == [
        (r.id, round(s, 6)) for r, s in reference.search(query, limit=5)
    ]


def test_dimension_mismatch(client: Any) -> None:
    storage = _storage(client)
    storage.save([_rec("a")])
    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.save([_rec("b", embedding=[0.1, 0.2])])
    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.search([0.1, 0.2])
    with pytest.raises(EmbeddingDimensionMismatchError):
        storage.update(_rec("a", embedding=[0.1, 0.2]))
    reopened = ChromaStorage(client=client, collection_name=storage._collection_name)
    with pytest.raises(EmbeddingDimensionMismatchError):
        reopened.save([_rec("c", embedding=[0.1, 0.2])])


def test_a_dimension_written_first_by_another_client_is_a_mismatch(client: Any) -> None:
    storage = _storage(client, vector_dim=None)
    client.create_collection(storage._collection_name, embedding_function=None).add(
        ids=["other"], embeddings=[[1.0, 0.0, 0.0]]
    )
    with pytest.raises(EmbeddingDimensionMismatchError) as raised:
        storage.save([_rec("mine")])
    assert (raised.value.stored_dim, raised.value.new_dim) == (3, 4)


def test_scans_are_capped_where_lancedb_caps_them(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chroma_storage, "_SCAN_ROWS_LIMIT", 3)
    storage = _storage(client)
    storage.save([_rec(f"r{i}", scope="/s", categories=["c"]) for i in range(5)])
    assert storage.count("/s") == 3
    assert storage.list_categories("/s") == {"c": 3}
    assert storage.count() == 5
    assert len(storage.list_records("/s")) == 5
    assert storage.delete(scope_prefix="/s") == 5


def test_memory_remember_recall_with_chroma(client: Any) -> None:
    from crewai.memory.unified_memory import Memory

    memory = Memory(
        storage=_storage(client, vector_dim=None),
        llm=MagicMock(),
        embedder=MagicMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts]),
    )
    memory.remember("We decided to use Python.", scope="/project", categories=["decision"], importance=0.7)
    matches = memory.recall("Python decision", scope="/project", limit=5, depth="shallow")
    assert [m.record.content for m in matches] == ["We decided to use Python."]
    assert memory.recall("Python decision", scope="/other", limit=5, depth="shallow") == []


# Random operations on LanceDBStorage and on ChromaStorage, with each result compared.
# The operations avoid the cases where the two differ on purpose, tested above: no
# record in the root scope, no scope name that starts like another one, no scope
# without its leading slash, no delete of a scope with older_than, no search by
# category. Every record has an embedding, so that no two distances are the same.
ORACLE_SCOPES = ["/a", "/a/b", "/a/c", "/b", "/b/a", "/c/a/b"]
ORACLE_QUERY_SCOPES = [*ORACLE_SCOPES, None, "/", "/c", "/c/a", "/a/"]


def _oracle_record(rng: random.Random, n: int) -> MemoryRecord:
    return MemoryRecord(
        id=f"r{n}",
        content=f"memory {n}",
        scope=rng.choice(ORACLE_SCOPES),
        categories=rng.sample(["x", "y", "z"], rng.randint(0, 2)),
        metadata={"k": rng.randint(0, 2)} if rng.random() < 0.7 else {},
        importance=round(rng.random(), 3),
        created_at=datetime(2026, 1, 1)
        + timedelta(seconds=rng.randint(0, 10**6), microseconds=rng.randint(1, 999_999)),
        last_accessed=datetime(2026, 1, 1),
        embedding=[rng.gauss(0, 1) for _ in range(8)],
        source=rng.choice([None, "u1"]),
        private=rng.random() < 0.2,
    )


def _same(a: MemoryRecord | None, b: MemoryRecord | None) -> None:
    assert (a is None) == (b is None)
    if a is not None and b is not None:
        assert a.model_dump() == b.model_dump()
        assert all(math.isclose(x, y, abs_tol=1e-6) for x, y in zip(a.embedding or [], b.embedding or [], strict=True))


@pytest.mark.parametrize("seed", range(4))
def test_random_operations_give_the_results_of_lancedb(client: Any, tmp_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    reference = LanceDBStorage(path=tmp_path / "lancedb", vector_dim=8, compact_every=0)
    ours = _storage(client, vector_dim=8)
    ids: list[str] = []
    for step in range(100):
        op = rng.choices(
            ["save", "search", "get", "list", "count", "info", "scopes", "categories", "delete", "update"],
            weights=[25, 25, 5, 10, 5, 5, 5, 5, 10, 5],
        )[0]
        where = f"step {step}, {op}"
        if op == "save":
            batch = [_oracle_record(rng, len(ids) + i) for i in range(rng.randint(1, 6))]
            reference.save(batch)
            ours.save(batch)
            ids.extend(r.id for r in batch)
        elif op == "search":
            kwargs: dict[str, Any] = {
                "scope_prefix": rng.choice(ORACLE_QUERY_SCOPES),
                "metadata_filter": {"k": rng.randint(0, 2)} if rng.random() < 0.3 else None,
                "limit": rng.randint(1, 12),
                "min_score": rng.choice([0.0, 0.02]),
            }
            query = [rng.gauss(0, 1) for _ in range(8)]
            expected = reference.search(query, **kwargs)
            got = ours.search(query, **kwargs)
            assert [r.id for r, _ in got] == [r.id for r, _ in expected], where
            for (ra, sa), (rb, sb) in zip(expected, got, strict=True):
                assert math.isclose(sa, sb, rel_tol=1e-5), where
                _same(ra, rb)
        elif op == "get":
            record_id = rng.choice([*ids, "missing"])
            _same(reference.get_record(record_id), ours.get_record(record_id))
        elif op == "list":
            kwargs = {"scope_prefix": rng.choice(ORACLE_QUERY_SCOPES), "limit": rng.randint(1, 30), "offset": rng.randint(0, 10)}
            assert [r.id for r in ours.list_records(**kwargs)] == [r.id for r in reference.list_records(**kwargs)], where
        elif op == "count":
            scope = rng.choice(ORACLE_QUERY_SCOPES)
            assert ours.count(scope) == reference.count(scope), where
        elif op == "info":
            scope = rng.choice([s for s in ORACLE_QUERY_SCOPES if s is not None])
            assert ours.get_scope_info(scope) == reference.get_scope_info(scope), where
        elif op == "scopes":
            parent = rng.choice(["/", "/a", "/b", "/c", "/c/a", "/a/"])
            assert ours.list_scopes(parent) == reference.list_scopes(parent), where
        elif op == "categories":
            scope = rng.choice(ORACLE_QUERY_SCOPES)
            assert ours.list_categories(scope) == reference.list_categories(scope), where
        elif op == "delete":
            kind = rng.choice(["ids", "scope", "categories", "metadata", "older_than"])
            cutoff = datetime(2026, 1, 1) + timedelta(seconds=rng.randint(0, 10**6))
            if kind == "ids":
                kwargs = {"record_ids": [*rng.sample(ids, min(len(ids), 2)), "missing"]}
            elif kind == "scope":
                kwargs = {"scope_prefix": rng.choice(ORACLE_SCOPES)}
            elif kind == "categories":
                kwargs = {"categories": [rng.choice(["x", "y", "z"])], "scope_prefix": rng.choice(ORACLE_QUERY_SCOPES)}
            elif kind == "metadata":
                kwargs = {"metadata_filter": {"k": rng.randint(0, 2)}, "scope_prefix": rng.choice(ORACLE_QUERY_SCOPES)}
            else:
                kwargs = {"older_than": cutoff}
            assert ours.delete(**kwargs) == reference.delete(**kwargs), f"{where} {kwargs}"
        elif ids:
            original = reference.get_record(rng.choice(ids))
            if original is not None:
                changed = original.model_copy(
                    update={"content": original.content + "!", "scope": rng.choice(ORACLE_SCOPES), "embedding": [rng.gauss(0, 1) for _ in range(8)]}
                )
                reference.update(changed)
                ours.update(changed)
    assert [r.id for r in ours.list_records(limit=10_000)] == [r.id for r in reference.list_records(limit=10_000)]
