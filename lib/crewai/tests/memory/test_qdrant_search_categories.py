"""Regression tests for server-side category filtering in QdrantEdgeStorage.search().

Categories were applied only AFTER the ANN fetch (limit*3 oversample), so
matching records ranked below the oversampled fetch were silently missed.
Categories now ride the server-side Filter; metadata_filter keeps the
client-side post-filter with oversampling.
"""

import importlib.util
from pathlib import Path

import pytest

from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage
from crewai.memory.types import MemoryRecord

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qdrant_edge") is None,
    reason="qdrant-edge-py not installed",
)

DIM = 4
NEAR = [1.0, 0.0, 0.0, 0.0]
FAR = [0.0, 0.0, 0.0, 1.0]


def _record(rid: str, categories: list[str], metadata: dict | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=rid,
        content=f"content {rid}",
        scope="/proj",
        categories=categories,
        metadata=metadata or {},
        embedding=FAR,
    )


def _storage(tmp_path: Path) -> QdrantEdgeStorage:
    return QdrantEdgeStorage(path=str(tmp_path / "edge"), vector_dim=DIM)


def test_matches_below_ann_fetch_limit_are_found(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    records = [
        MemoryRecord(
            id=f"noise-{i}",
            content=f"noise {i}",
            scope="/proj",
            categories=["other"],
            embedding=NEAR,
        )
        for i in range(8)
    ]
    records += [_record(f"target-{i}", ["target"]) for i in range(2)]
    storage.save(records)

    hits = storage.search(NEAR, categories=["target"], limit=2)

    ids = sorted(record.id for record, _ in hits)
    assert ids == ["target-0", "target-1"]


def test_categories_limit_respected(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    records = [_record(f"target-{i}", ["target"]) for i in range(5)]
    storage.save(records)

    hits = storage.search(FAR, categories=["target"], limit=2)

    assert len(hits) == 2
    assert all(record.id.startswith("target-") for record, _ in hits)


def test_metadata_filter_with_categories_still_applies(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    records = [
        _record("target-a", ["target"], metadata={"kind": "keep"}),
        _record("target-b", ["target"], metadata={"kind": "drop"}),
        _record("noise", ["other"], metadata={"kind": "keep"}),
    ]
    storage.save(records)

    hits = storage.search(
        FAR,
        categories=["target"],
        metadata_filter={"kind": "keep"},
        limit=5,
    )

    assert [record.id for record, _ in hits] == ["target-a"]
