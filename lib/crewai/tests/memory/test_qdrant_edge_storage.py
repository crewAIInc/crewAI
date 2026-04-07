"""Tests for Qdrant Edge storage backend."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qdrant_edge") is None,
    reason="qdrant-edge-py not installed",
)

if TYPE_CHECKING:
    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

from crewai.memory.types import MemoryRecord


def _make_storage(path: str, vector_dim: int = 4) -> QdrantEdgeStorage:
    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

    return QdrantEdgeStorage(path=path, vector_dim=vector_dim)


@pytest.fixture
def storage(tmp_path: Path) -> QdrantEdgeStorage:
    return _make_storage(str(tmp_path / "edge"))


def _rec(
    content: str = "test",
    scope: str = "/",
    categories: list[str] | None = None,
    importance: float = 0.5,
    embedding: list[float] | None = None,
    metadata: dict | None = None,
    created_at: datetime | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        scope=scope,
        categories=categories or [],
        importance=importance,
        embedding=embedding or [0.1, 0.2, 0.3, 0.4],
        metadata=metadata or {},
        **({"created_at": created_at} if created_at else {}),
    )


# --- Basic CRUD ---


def test_save_search(storage: QdrantEdgeStorage) -> None:
    r = _rec(content="test content", scope="/foo", categories=["cat1"], importance=0.8)
    storage.save([r])
    results = storage.search([0.1, 0.2, 0.3, 0.4], scope_prefix="/foo", limit=5)
    assert len(results) == 1
    rec, score = results[0]
    assert rec.content == "test content"
    assert rec.scope == "/foo"
    assert score >= 0.0


def test_delete_count(storage: QdrantEdgeStorage) -> None:
    r = _rec(scope="/")
    storage.save([r])
    assert storage.count() == 1
    n = storage.delete(scope_prefix="/")
    assert n >= 1
    assert storage.count() == 0


def test_update_get_record(storage: QdrantEdgeStorage) -> None:
    r = _rec(content="original", scope="/a")
    storage.save([r])
    r.content = "updated"
    storage.update(r)
    found = storage.get_record(r.id)
    assert found is not None
    assert found.content == "updated"


def test_get_record_not_found(storage: QdrantEdgeStorage) -> None:
    assert storage.get_record("nonexistent-id") is None


# --- Scope operations ---


def test_list_scopes_get_scope_info(storage: QdrantEdgeStorage) -> None:
    storage.save([
        _rec(content="a", scope="/"),
        _rec(content="b", scope="/team"),
    ])
    scopes = storage.list_scopes("/")
    assert "/team" in scopes
    info = storage.get_scope_info("/")
    assert info.record_count >= 1
    assert info.path == "/"


def test_scope_prefix_filter(storage: QdrantEdgeStorage) -> None:
    storage.save([
        _rec(content="sales note", scope="/crew/sales"),
        _rec(content="eng note", scope="/crew/eng"),
        _rec(content="other note", scope="/other"),
    ])
    results = storage.search([0.1, 0.2, 0.3, 0.4], scope_prefix="/crew", limit=10)
    assert len(results) == 2
    scopes = {r.scope for r, _ in results}
    assert "/crew/sales" in scopes
    assert "/crew/eng" in scopes


# --- Filtering ---


def test_category_filter(storage: QdrantEdgeStorage) -> None:
    storage.save([
        _rec(content="cat1 item", categories=["cat1"]),
        _rec(content="cat2 item", categories=["cat2"]),
    ])
    results = storage.search(
        [0.1, 0.2, 0.3, 0.4], categories=["cat1"], limit=10
    )
    assert len(results) == 1
    assert results[0][0].categories == ["cat1"]


def test_metadata_filter(storage: QdrantEdgeStorage) -> None:
    storage.save([
        _rec(content="with key", metadata={"env": "prod"}),
        _rec(content="without key", metadata={"env": "dev"}),
    ])
    results = storage.search(
        [0.1, 0.2, 0.3, 0.4], metadata_filter={"env": "prod"}, limit=10
    )
    assert len(results) == 1
    assert results[0][0].metadata["env"] == "prod"


# --- List & pagination ---


def test_list_records_pagination(storage: QdrantEdgeStorage) -> None:
    records = [
        _rec(
            content=f"item {i}",
            created_at=datetime(2025, 1, 1) + timedelta(days=i),
        )
        for i in range(5)
    ]
    storage.save(records)
    page1 = storage.list_records(limit=2, offset=0)
    page2 = storage.list_records(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    # Newest first.
    assert page1[0].created_at >= page1[1].created_at


def test_list_categories(storage: QdrantEdgeStorage) -> None:
    storage.save([
        _rec(categories=["a", "b"]),
        _rec(categories=["b", "c"]),
    ])
    cats = storage.list_categories()
    assert cats.get("b", 0) == 2
    assert cats.get("a", 0) >= 1
    assert cats.get("c", 0) >= 1


# --- Touch & reset ---


def test_touch_records(storage: QdrantEdgeStorage) -> None:
    r = _rec()
    storage.save([r])
    before = storage.get_record(r.id)
    assert before is not None
    old_accessed = before.last_accessed
    storage.touch_records([r.id])
    after = storage.get_record(r.id)
    assert after is not None
    assert after.last_accessed >= old_accessed


def test_reset_full(storage: QdrantEdgeStorage) -> None:
    storage.save([_rec(scope="/a"), _rec(scope="/b")])
    assert storage.count() == 2
    storage.reset()
    assert storage.count() == 0


def test_reset_scoped(storage: QdrantEdgeStorage) -> None:
    storage.save([_rec(scope="/a"), _rec(scope="/b")])
    storage.reset(scope_prefix="/a")
    assert storage.count() == 1


# --- Dual-shard & sync ---


def test_flush_to_central(tmp_path: Path) -> None:
    s = _make_storage(str(tmp_path / "edge"))
    s.save([_rec(content="to sync")])
    assert s._local_has_data
    s.flush_to_central()
    assert not s._local_has_data
    assert not s._local_path.exists()
    # Central should have the record.
    assert s.count() == 1


def test_dual_shard_search(tmp_path: Path) -> None:
    s = _make_storage(str(tmp_path / "edge"))
    # Save and flush to central.
    s.save([_rec(content="central record", scope="/a")])
    s.flush_to_central()
    # Save to local only.
    s._closed = False  # Reset for continued use.
    s.save([_rec(content="local record", scope="/b")])
    # Search should find both.
    results = s.search([0.1, 0.2, 0.3, 0.4], limit=10)
    assert len(results) == 2
    contents = {r.content for r, _ in results}
    assert "central record" in contents
    assert "local record" in contents


def test_close_lifecycle(tmp_path: Path) -> None:
    s = _make_storage(str(tmp_path / "edge"))
    s.save([_rec(content="persisted")])
    s.close()
    # Reopen a new storage — should find the record in central.
    s2 = _make_storage(str(tmp_path / "edge"))
    results = s2.search([0.1, 0.2, 0.3, 0.4], limit=5)
    assert len(results) == 1
    assert results[0][0].content == "persisted"
    s2.close()


def test_orphaned_shard_cleanup(tmp_path: Path) -> None:
    base = tmp_path / "edge"
    # Create a fake orphaned shard using a PID that doesn't exist.
    fake_pid = 99999999
    s1 = _make_storage(str(base))
    # Manually create a shard at the orphaned path.
    orphan_path = base / f"worker-{fake_pid}"
    orphan_path.mkdir(parents=True, exist_ok=True)
    from qdrant_edge import (
        EdgeConfig,
        EdgeShard,
        EdgeVectorParams,
        Distance,
        Point,
        UpdateOperation,
    )

    config = EdgeConfig(
        vectors={"memory": EdgeVectorParams(size=4, distance=Distance.Cosine)}
    )
    orphan = EdgeShard.create(str(orphan_path), config)
    orphan.update(
        UpdateOperation.upsert_points([
            Point(
                id=12345,
                vector={"memory": [0.5, 0.5, 0.5, 0.5]},
                payload={
                    "record_id": "orphan-uuid",
                    "content": "orphaned",
                    "scope": "/",
                    "scope_ancestors": ["/"],
                    "categories": [],
                    "metadata": {},
                    "importance": 0.5,
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                    "last_accessed": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                    "source": "",
                    "private": False,
                },
            )
        ])
    )
    orphan.flush()
    orphan.close()
    s1.close()

    # Creating a new storage should detect and recover the orphaned shard.
    s2 = _make_storage(str(base))
    assert not orphan_path.exists()
    # The orphaned record should now be in central.
    results = s2.search([0.5, 0.5, 0.5, 0.5], limit=5)
    assert len(results) >= 1
    assert any(r.content == "orphaned" for r, _ in results)
    s2.close()


# --- Integration with Memory class ---


def test_memory_with_qdrant_edge(tmp_path: Path) -> None:
    from crewai.memory.unified_memory import Memory

    mock_embedder = MagicMock()
    mock_embedder.side_effect = lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    storage = _make_storage(str(tmp_path / "edge"))
    m = Memory(
        storage=storage,
        llm=MagicMock(),
        embedder=mock_embedder,
    )
    r = m.remember(
        "We decided to use Qdrant Edge.",
        scope="/project",
        categories=["decision"],
        importance=0.7,
    )
    assert r.content == "We decided to use Qdrant Edge."

    matches = m.recall("Qdrant", scope="/project", limit=5, depth="shallow")
    assert len(matches) >= 1
    m.close()


def test_memory_string_storage_qdrant_edge(tmp_path: Path) -> None:
    """Test that storage='qdrant-edge' string instantiation works."""
    import os

    os.environ["CREWAI_STORAGE_DIR"] = str(tmp_path)
    try:
        from crewai.memory.unified_memory import Memory

        mock_embedder = MagicMock()
        mock_embedder.side_effect = lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        m = Memory(
            storage="qdrant-edge",
            llm=MagicMock(),
            embedder=mock_embedder,
        )
        from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

        assert isinstance(m._storage, QdrantEdgeStorage)
        m.close()
    finally:
        os.environ.pop("CREWAI_STORAGE_DIR", None)
