"""Regression tests for tz-aware older_than thresholds in delete().

MemoryRecord.created_at defaults to naive UTC, so a caller passing a tz-aware
datetime (e.g. datetime.now(timezone.utc)) previously raised TypeError in the
lancedb categories path and was silently swallowed (0 records deleted) in the
qdrant path. Both backends now normalise aware thresholds to naive UTC.
"""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord

DIM = 4
AWARE_YESTERDAY = datetime.now(timezone.utc) - timedelta(days=1)


def _old_record(rid: str) -> MemoryRecord:
    record = MemoryRecord(
        id=rid,
        content="old note",
        scope="/proj",
        categories=["note"],
        embedding=[0.1] * DIM,
    )
    record.created_at = datetime.utcnow() - timedelta(days=30)
    return record


def test_lancedb_categories_aware_threshold_deletes_old_records(
    tmp_path: Path,
) -> None:
    storage = LanceDBStorage(path=tmp_path, vector_dim=DIM, compact_every=0)
    storage.save([_old_record("old")])

    deleted = storage.delete(categories=["note"], older_than=AWARE_YESTERDAY)

    assert deleted == 1
    assert storage.count() == 0


def test_lancedb_categories_aware_threshold_keeps_new_records(
    tmp_path: Path,
) -> None:
    storage = LanceDBStorage(path=tmp_path, vector_dim=DIM, compact_every=0)
    storage.save(
        [MemoryRecord(id="fresh", content="fresh", scope="/proj", embedding=[0.1] * DIM)]
    )

    deleted = storage.delete(categories=["note"], older_than=AWARE_YESTERDAY)

    assert deleted == 0
    assert storage.count() == 1


def test_lancedb_sql_path_aware_offset_matches_utc_instant(tmp_path: Path) -> None:
    """The same instant expressed with a non-UTC offset must delete the same window."""
    threshold = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    est_threshold = threshold.astimezone(timezone(timedelta(hours=-5)))
    assert est_threshold.timestamp() == threshold.timestamp()

    def make_storage() -> LanceDBStorage:
        storage = LanceDBStorage(path=tmp_path / f"db-{uuid4().hex}", vector_dim=DIM, compact_every=0)
        record = MemoryRecord(
            id="rec", content="note", scope="/proj", embedding=[0.1] * DIM
        )
        # 2h before the threshold instant, naive UTC
        record.created_at = datetime(2026, 1, 1, 10, 0, 0)
        storage.save([record])
        return storage

    utc_storage = make_storage()
    est_storage = make_storage()

    n_utc = utc_storage.delete(scope_prefix="/proj", older_than=threshold)
    n_est = est_storage.delete(scope_prefix="/proj", older_than=est_threshold)

    assert n_utc == n_est == 1


qdrant_available = importlib.util.find_spec("qdrant_edge") is not None
pytestmark_qdrant = pytest.mark.skipif(
    not qdrant_available, reason="qdrant-edge-py not installed"
)


@pytestmark_qdrant
def test_qdrant_categories_aware_threshold_deletes_old_records(
    tmp_path: Path,
) -> None:
    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

    storage = QdrantEdgeStorage(path=str(tmp_path / "edge"), vector_dim=DIM)
    storage.save([_old_record("old")])

    deleted = storage.delete(categories=["note"], older_than=AWARE_YESTERDAY)

    assert deleted == 1


@pytestmark_qdrant
def test_qdrant_record_ids_aware_threshold_keeps_new_records(
    tmp_path: Path,
) -> None:
    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

    storage = QdrantEdgeStorage(path=str(tmp_path / "edge"), vector_dim=DIM)
    storage.save(
        [MemoryRecord(id="fresh", content="fresh", scope="/proj", embedding=[0.1] * DIM)]
    )

    deleted = storage.delete(record_ids=["fresh"], older_than=AWARE_YESTERDAY)

    assert deleted == 0
