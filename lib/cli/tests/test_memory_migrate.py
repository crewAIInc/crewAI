"""Tests for `crewai memory migrate` -- the per-tenant migration command.

These tests are hermetic: they create a fresh LanceDB table per test and
operate on it via the underlying lancedb package directly, never crossing
network boundaries. The Memory subsystem is not involved -- the migrate
command is a pure schema/data transformation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import lancedb  # type: ignore[import-untyped]
import pytest

from crewai_cli.memory_migrate import run_migrate


def _make_pre_isolation_table(path: Path, table_name: str = "memories") -> None:
    """Create a LanceDB table that looks like a pre-isolation install.

    Schema mirrors LanceDBStorage._create_table BEFORE PR #1, so it has all
    the legacy columns but no tenant_id / user_id. This is what the migrate
    command must be able to handle.
    """
    path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(path))
    rows = [
        {
            "id": "row-1",
            "content": "alice's note",
            "scope": "/",
            "categories_str": "[]",
            "metadata_str": json.dumps({"customer_id": "acme"}),
            "importance": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "source": "",
            "private": False,
            "vector": [0.1, 0.2, 0.3, 0.4],
        },
        {
            "id": "row-2",
            "content": "bob's note",
            "scope": "/",
            "categories_str": "[]",
            "metadata_str": json.dumps({"customer_id": "globex"}),
            "importance": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "source": "",
            "private": False,
            "vector": [0.1, 0.2, 0.3, 0.4],
        },
        {
            "id": "row-3",
            "content": "untagged note",
            "scope": "/",
            "categories_str": "[]",
            "metadata_str": "{}",
            "importance": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "source": "",
            "private": False,
            "vector": [0.1, 0.2, 0.3, 0.4],
        },
    ]
    db.create_table(table_name, rows)


def test_migrate_on_missing_directory_is_noop(tmp_path: Path) -> None:
    summary = run_migrate(
        storage_dir=str(tmp_path / "nonexistent"),
        default_tenant="_default",
        from_metadata_key=None,
        table_name="memories",
        dry_run=False,
    )
    assert summary["rows_scanned"] == 0
    assert summary["rows_to_stamp"] == 0
    assert summary["rows_updated"] == 0


def test_migrate_on_missing_table_is_noop(tmp_path: Path) -> None:
    (tmp_path / "memory").mkdir()
    summary = run_migrate(
        storage_dir=str(tmp_path / "memory"),
        default_tenant="_default",
        from_metadata_key=None,
        table_name="memories",
        dry_run=False,
    )
    assert summary["rows_scanned"] == 0


def test_migrate_adds_columns_with_default_tenant(tmp_path: Path) -> None:
    store = tmp_path / "memory"
    _make_pre_isolation_table(store)

    summary = run_migrate(
        storage_dir=str(store),
        default_tenant="_default",
        from_metadata_key=None,
        table_name="memories",
        dry_run=False,
    )

    # The column was added with default '_default' so every row has a
    # tenant_id matching default_tenant; no per-row update needed.
    assert summary["rows_scanned"] == 3
    assert summary["rows_to_stamp"] == 0
    assert summary["rows_updated"] == 0

    # Verify the schema migration happened: tenant_id column exists.
    db = lancedb.connect(str(store))
    table = db.open_table("memories")
    field_names = {f.name for f in table.schema}
    assert "tenant_id" in field_names
    assert "user_id" in field_names


def test_migrate_with_metadata_key_stamps_per_row(tmp_path: Path) -> None:
    store = tmp_path / "memory"
    _make_pre_isolation_table(store)

    summary = run_migrate(
        storage_dir=str(store),
        default_tenant="_default",
        from_metadata_key="customer_id",
        table_name="memories",
        dry_run=False,
    )

    assert summary["rows_scanned"] == 3
    # Rows 1 and 2 have customer_id; row 3 does not.
    assert summary["rows_with_metadata_key"] == 2
    # Rows 1 and 2 need a non-default tenant; row 3 stays '_default'.
    assert summary["rows_to_stamp"] == 2
    assert summary["rows_updated"] == 2

    db = lancedb.connect(str(store))
    table = db.open_table("memories")
    by_id = {row["id"]: row for row in table.search().to_list()}
    assert by_id["row-1"]["tenant_id"] == "acme"
    assert by_id["row-2"]["tenant_id"] == "globex"
    assert by_id["row-3"]["tenant_id"] == "_default"


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    store = tmp_path / "memory"
    _make_pre_isolation_table(store)

    first = run_migrate(
        storage_dir=str(store),
        default_tenant="_default",
        from_metadata_key="customer_id",
        table_name="memories",
        dry_run=False,
    )
    second = run_migrate(
        storage_dir=str(store),
        default_tenant="_default",
        from_metadata_key="customer_id",
        table_name="memories",
        dry_run=False,
    )

    assert first["rows_updated"] == 2
    assert second["rows_scanned"] == 3
    # Second pass finds nothing to change.
    assert second["rows_to_stamp"] == 0
    assert second["rows_updated"] == 0


def test_migrate_dry_run_does_not_write(tmp_path: Path) -> None:
    store = tmp_path / "memory"
    _make_pre_isolation_table(store)

    summary = run_migrate(
        storage_dir=str(store),
        default_tenant="_default",
        from_metadata_key="customer_id",
        table_name="memories",
        dry_run=True,
    )

    assert summary["rows_to_stamp"] == 2
    assert summary["rows_updated"] == 0

    # Verify the table was not modified: the new columns were NOT added
    # because dry_run skipped the add_columns call.
    db = lancedb.connect(str(store))
    table = db.open_table("memories")
    field_names = {f.name for f in table.schema}
    assert "tenant_id" not in field_names


def test_migrate_pagination_streams_past_page_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tables larger than a single page must be fully scanned, not truncated.

    Pre-fix bug: the migrator did ``.limit(10_000_000).to_list()`` and silently
    dropped anything past that cap. The paginated streamer must keep going
    until an empty page is returned.
    """
    from crewai_cli import memory_migrate

    # Force a small page size so a small fixture exercises the loop.
    monkeypatch.setattr(memory_migrate, "_SCAN_PAGE_SIZE", 2)

    store = tmp_path / "memory"
    store.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(store))
    rows = [
        {
            "id": f"row-{i}",
            "content": f"item {i}",
            "scope": "/",
            "categories_str": "[]",
            "metadata_str": json.dumps({"customer_id": f"cust-{i}"}),
            "importance": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "source": "",
            "private": False,
            "vector": [0.1, 0.2, 0.3, 0.4],
        }
        for i in range(7)  # 4 pages: 2, 2, 2, 1
    ]
    db.create_table("memories", rows)

    summary = memory_migrate.run_migrate(
        storage_dir=str(store),
        default_tenant="_default",
        from_metadata_key="customer_id",
        table_name="memories",
        dry_run=False,
    )

    # All 7 rows must be visited even though the page size is 2.
    assert summary["rows_scanned"] == 7
    assert summary["rows_with_metadata_key"] == 7
    assert summary["rows_updated"] == 7

    by_id = {
        r["id"]: r
        for r in db.open_table("memories").search().to_list()
    }
    for i in range(7):
        assert by_id[f"row-{i}"]["tenant_id"] == f"cust-{i}"


def test_migrate_rejects_empty_default_tenant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="default_tenant"):
        run_migrate(
            storage_dir=str(tmp_path),
            default_tenant="",
            from_metadata_key=None,
            table_name="memories",
            dry_run=False,
        )
