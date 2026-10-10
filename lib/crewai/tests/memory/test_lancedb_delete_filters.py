"""Regression tests for intersecting LanceDB memory deletion filters."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
import pytest


CUTOFF = datetime(2026, 1, 15)


@pytest.fixture
def storage(tmp_path: Path) -> LanceDBStorage:
    return LanceDBStorage(path=tmp_path / "memory", vector_dim=4, compact_every=0)


@pytest.fixture
def records(storage: LanceDBStorage) -> list[MemoryRecord]:
    specs = [
        ("target", "/tenant1", "finance", "active", -1),
        ("unrequested", "/tenant1", "finance", "active", -1),
        ("foreign", "/tenant2", "finance", "active", -1),
        ("global", "/", "finance", "active", -1),
        ("sibling", "/tenant10", "finance", "active", -1),
        ("child", "/tenant1/child", "finance", "active", -1),
        ("boundary", "/tenant1", "finance", "active", 0),
        ("newer", "/tenant1", "finance", "active", 1),
        ("category-mismatch", "/tenant1", "engineering", "active", -1),
        ("metadata-mismatch", "/tenant1", "finance", "archived", -1),
    ]
    result = [
        MemoryRecord(
            id=record_id,
            content=record_id,
            scope=scope,
            categories=[category],
            metadata={"status": status},
            created_at=CUTOFF + timedelta(days=age),
            embedding=[0.0] * 4,
        )
        for record_id, scope, category, status, age in specs
    ]
    storage.save(result)
    return result


@pytest.mark.parametrize(
    ("filters", "deleted_ids"),
    [
        pytest.param(
            {
                "record_ids": ["target", "foreign", "global", "sibling", "child"],
                "scope_prefix": "/tenant1",
            },
            {"target", "child"},
            id="ids-and-scope",
        ),
        pytest.param(
            {"record_ids": ["target", "boundary", "newer"], "older_than": CUTOFF},
            {"target"},
            id="ids-and-age",
        ),
        pytest.param(
            {
                "record_ids": ["target", "category-mismatch"],
                "categories": ["finance", "other"],
            },
            {"target"},
            id="ids-and-any-category",
        ),
        pytest.param(
            {
                "record_ids": ["target", "metadata-mismatch"],
                "metadata_filter": {"status": "active"},
            },
            {"target"},
            id="ids-and-metadata",
        ),
        pytest.param(
            {
                "record_ids": [
                    "target",
                    "foreign",
                    "global",
                    "sibling",
                    "boundary",
                    "newer",
                    "category-mismatch",
                    "metadata-mismatch",
                ],
                "scope_prefix": "/tenant1",
                "older_than": CUTOFF,
                "categories": ["finance"],
                "metadata_filter": {"status": "active"},
            },
            {"target"},
            id="all-filters",
        ),
        pytest.param(
            {"scope_prefix": "/tenant1", "older_than": CUTOFF},
            {
                "target",
                "unrequested",
                "child",
                "category-mismatch",
                "metadata-mismatch",
            },
            id="scope-and-age",
        ),
        pytest.param(
            {
                "scope_prefix": "tenant1/",
                "categories": ["finance"],
                "older_than": CUTOFF,
            },
            {"target", "unrequested", "child", "metadata-mismatch"},
            id="normalized-scope-and-category",
        ),
        pytest.param(
            {
                "record_ids": ["target"],
                "metadata_filter": {"status": "active", "missing": True},
            },
            set(),
            id="all-metadata-keys-required",
        ),
        pytest.param(
            {"record_ids": ["missing"], "categories": ["finance"]},
            set(),
            id="no-matching-id",
        ),
    ],
)
def test_delete_intersects_filters(
    storage: LanceDBStorage,
    records: list[MemoryRecord],
    filters: dict[str, Any],
    deleted_ids: set[str],
) -> None:
    assert storage.delete(**filters) == len(deleted_ids)
    assert {
        record.id for record in records if storage.get_record(record.id) is None
    } == deleted_ids


@pytest.mark.parametrize("categories", [None, ["finance"]])
def test_delete_quoted_ids(
    storage: LanceDBStorage, categories: list[str] | None
) -> None:
    record_id = "record'quoted"
    storage.save(
        [
            MemoryRecord(
                id=rid, content=rid, categories=["finance"], embedding=[0.0] * 4
            )
            for rid in [record_id, "keep"]
        ]
    )

    assert storage.delete(record_ids=[record_id], categories=categories) == 1
    assert storage.get_record(record_id) is None
    assert storage.get_record("keep") is not None


@pytest.mark.parametrize(
    "filters", [{"categories": ["finance"]}, {"metadata_filter": {"status": "active"}}]
)
def test_delete_filtered_empty_id(
    storage: LanceDBStorage, filters: dict[str, Any]
) -> None:
    storage.save(
        [
            MemoryRecord(
                id="",
                content="x",
                categories=["finance"],
                metadata={"status": "active"},
                embedding=[0.0] * 4,
            )
        ]
    )

    assert storage.delete(**filters) == 1
    assert storage.get_record("") is None


@pytest.mark.parametrize("categories", [None, ["finance"]])
def test_delete_scope_is_literal(
    storage: LanceDBStorage, categories: list[str] | None
) -> None:
    scope = "/team's_100%"
    storage.save(
        [
            MemoryRecord(
                id=str(i),
                content="x",
                scope=value,
                categories=["finance"],
                embedding=[0.0] * 4,
            )
            for i, value in enumerate(
                [scope, scope + "/child", "/team'sX100other", "/"]
            )
        ]
    )

    assert storage.delete(scope_prefix=scope, categories=categories) == 2
    assert storage.get_record("2") is not None
    assert storage.get_record("3") is not None


@pytest.mark.parametrize("categories", [None, ["finance"]])
def test_delete_compares_timestamp_instants(
    storage: LanceDBStorage, categories: list[str] | None
) -> None:
    timestamps = {
        "older": "2026-01-15T01:00:00+02:00",
        "equal": "2026-01-14T22:00:00-02:00",
        "newer": "2026-01-14T23:00:00-02:00",
    }
    storage.save(
        [
            MemoryRecord(
                id=record_id,
                content=record_id,
                categories=["finance"],
                created_at=datetime.fromisoformat(timestamp),
                embedding=[0.0] * 4,
            )
            for record_id, timestamp in timestamps.items()
        ]
    )

    assert (
        storage.delete(
            record_ids=list(timestamps),
            older_than=datetime.fromisoformat("2026-01-15T00:00:00+00:00"),
            categories=categories,
        )
        == 1
    )
    assert storage.get_record("older") is None
    assert storage.get_record("equal") is not None
    assert storage.get_record("newer") is not None


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"scope_prefix": "/"},
        {"record_ids": [], "categories": [], "metadata_filter": {}},
    ],
)
def test_delete_without_filters_keeps_existing_behavior(
    storage: LanceDBStorage, records: list[MemoryRecord], filters: dict[str, Any]
) -> None:
    assert storage.delete(**filters) == len(records)
    assert storage.count() == 0
    assert storage.delete(**filters) == 0


@pytest.mark.parametrize(
    ("filters", "deleted_count"),
    [
        ({"record_ids": ["50000"], "older_than": CUTOFF + timedelta(days=1)}, 1),
        ({"record_ids": ["50000"], "categories": ["finance"]}, 1),
        ({"categories": ["finance"]}, 50_001),
    ],
)
def test_delete_beyond_scan_limit(
    storage: LanceDBStorage, filters: dict[str, Any], deleted_count: int
) -> None:
    storage.save(
        [
            MemoryRecord(
                id=str(i),
                content="x",
                categories=["finance"],
                embedding=[0.0] * 4,
                created_at=CUTOFF,
                last_accessed=CUTOFF,
            )
            for i in range(50_001)
        ]
    )

    assert storage.delete(**filters) == deleted_count
    assert storage.get_record("50000") is None
    assert storage.count() == 50_001 - deleted_count


@pytest.mark.asyncio
async def test_adelete_intersects_filters(
    storage: LanceDBStorage, records: list[MemoryRecord]
) -> None:
    assert (
        await storage.adelete(
            scope_prefix="/tenant1",
            record_ids=["target", "foreign", "newer"],
            older_than=CUTOFF,
            categories=["finance"],
        )
        == 1
    )
    assert storage.get_record("foreign") is not None
    assert storage.get_record("newer") is not None
