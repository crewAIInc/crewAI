"""Logic for `crewai memory migrate` -- stamp unscoped LanceDB rows with a tenant_id.

The migration is two layers:

1. **Schema migration** (automatic on open). When LanceDBStorage opens an
   existing table, _ensure_tenant_columns() adds the tenant_id and user_id
   columns with default '_default' if they are missing. So the column exists
   after the first open, and every row reads back as '_default' until something
   explicitly rewrites it.

2. **Per-row migration** (this command). When --from-metadata-key is supplied,
   this command scans every row and copies the metadata[key] value into the
   tenant_id column. Rows missing the key keep the '_default' fallback.

The command is idempotent. Running it twice does not change anything that was
already correct.

The scan is paginated and column-selected: it never loads the ``vector`` column
(which dominates per-row memory) and never tries to materialize the whole table
at once. The previous implementation capped the read at 10_000_000 rows and
silently truncated past it; this one streams every row in fixed-size pages.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import os
from pathlib import Path
from typing import Any, TypedDict


_logger = logging.getLogger(__name__)


# Only the columns we actually inspect during migration. Crucially this omits
# the ``vector`` column -- on a 1536-dimension index, that column is ~6 KiB per
# row and dominates memory use. We do not need it to stamp tenant_id.
_SCAN_COLUMNS = ["id", "tenant_id", "metadata_str"]

# Page size for the paginated scan. 5_000 keeps peak memory bounded
# (~5 MiB per page including json metadata) while amortizing per-call overhead.
_SCAN_PAGE_SIZE = 5_000


class MigrateSummary(TypedDict):
    storage_dir: str
    table_name: str
    rows_scanned: int
    rows_to_stamp: int
    rows_with_metadata_key: int
    rows_updated: int


def _iter_rows_paginated(table: Any) -> Iterator[dict[str, Any]]:
    """Yield rows from a LanceDB table in fixed-size pages, columns-selected.

    Selects only columns we actually inspect during migration. The heavy
    ``vector`` column is never materialized. ``tenant_id`` is only selected
    when the schema already has it -- in dry-run against a pre-isolation
    table the column does not exist yet, and asking for it raises a LanceDB
    schema error. A missing tenant_id column is equivalent to every row
    being unstamped for the purpose of migration accounting.

    Pagination uses an offset cursor. The migration is read-then-write with
    no concurrent writers expected, so the per-row id stays stable across
    pages.
    """
    available = {field.name for field in table.schema}
    select_columns = [c for c in _SCAN_COLUMNS if c in available]
    offset = 0
    while True:
        query = table.search().select(select_columns).limit(_SCAN_PAGE_SIZE)
        # .offset() is the chained form on lancedb >= 0.16; older versions
        # only support reading from the start. The migration documents the
        # supported lancedb version range in pyproject; here we assume the
        # chain is available.
        if offset:
            query = query.offset(offset)
        page = query.to_list()
        if not page:
            return
        for row in page:
            yield row
        if len(page) < _SCAN_PAGE_SIZE:
            return
        offset += len(page)


def _resolve_storage_dir(storage_dir: str | None) -> Path:
    """Pick the storage directory the same way LanceDBStorage does.

    Priority:
        1. --storage-dir CLI flag
        2. $CREWAI_STORAGE_DIR/memory
        3. db_storage_path() / memory  (platform data dir)
    """
    if storage_dir:
        return Path(storage_dir)
    env_dir = os.environ.get("CREWAI_STORAGE_DIR")
    if env_dir:
        return Path(env_dir) / "memory"
    from crewai_core.paths import db_storage_path

    return Path(db_storage_path()) / "memory"


def run_migrate(
    *,
    storage_dir: str | None,
    default_tenant: str,
    from_metadata_key: str | None,
    table_name: str,
    dry_run: bool,
) -> MigrateSummary:
    """Run the migration and return a summary dict.

    Returns a dict the CLI prints; raises only on I/O or schema problems.
    """
    if not default_tenant:
        raise ValueError("default_tenant must be a non-empty string")

    resolved_dir = _resolve_storage_dir(storage_dir)
    summary: MigrateSummary = {
        "storage_dir": str(resolved_dir),
        "table_name": table_name,
        "rows_scanned": 0,
        "rows_to_stamp": 0,
        "rows_with_metadata_key": 0,
        "rows_updated": 0,
    }

    if not resolved_dir.exists():
        _logger.info(
            "Storage directory %s does not exist; nothing to migrate.", resolved_dir
        )
        return summary

    import lancedb  # type: ignore[import-untyped]

    db = lancedb.connect(str(resolved_dir))
    try:
        table = db.open_table(table_name)
    except Exception:
        _logger.info(
            "No table %r in %s; nothing to migrate.", table_name, resolved_dir
        )
        return summary

    # Opening the table via LanceDBStorage path would auto-add the columns;
    # here we use lancedb directly. Replicate the column add so this command
    # also fixes pre-isolation schemas without depending on LanceDBStorage init.
    existing_fields = {field.name for field in table.schema}
    to_add: dict[str, str] = {}
    if "tenant_id" not in existing_fields:
        to_add["tenant_id"] = f"'{default_tenant}'"
    if "user_id" not in existing_fields:
        to_add["user_id"] = "''"
    if to_add and not dry_run:
        try:
            table.add_columns(to_add)
        except Exception as exc:
            _logger.warning(
                "Could not add tenant columns to %r: %s. "
                "Continuing -- per-row updates will still attempt to set the field.",
                table_name,
                exc,
            )

    # Scan every row that needs stamping. The iterator pages through the
    # table fetching only id, tenant_id, and metadata_str -- the heavy
    # ``vector`` column is never materialized. A row needs stamping if:
    #   - tenant_id is missing/empty, OR
    #   - --from-metadata-key was provided AND metadata[key] differs from the
    #     row's current tenant_id (idempotent: re-runs don't re-update
    #     unchanged rows).
    to_update: list[dict[str, Any]] = []
    for row in _iter_rows_paginated(table):
        summary["rows_scanned"] += 1
        # A row that pre-dates the tenant_id column (or has an empty value)
        # is treated as if it had been stamped with default_tenant. This makes
        # dry-run and real-run report identical rows_to_stamp counts -- the
        # alternative is misleading dry-run output that overstates the change
        # because it doesn't simulate the column-add side effect.
        raw_tenant = (row.get("tenant_id") or "").strip()
        current_tenant = raw_tenant if raw_tenant else default_tenant
        target_tenant = default_tenant

        if from_metadata_key:
            metadata_str = row.get("metadata_str") or "{}"
            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                metadata = {}
            key_value = metadata.get(from_metadata_key)
            if key_value:
                target_tenant = str(key_value)
                summary["rows_with_metadata_key"] += 1

        if current_tenant == target_tenant:
            continue
        to_update.append({"id": row["id"], "tenant_id": target_tenant})

    summary["rows_to_stamp"] = len(to_update)

    if dry_run:
        return summary

    # Apply updates. LanceDB does row updates via merge_insert or per-row
    # update; using per-row table.update() with a WHERE clause is the
    # simplest correct path and works on every supported version.
    for entry in to_update:
        safe_id = str(entry["id"]).replace("'", "''")
        safe_tenant = str(entry["tenant_id"]).replace("'", "''")
        table.update(
            where=f"id = '{safe_id}'",
            values={"tenant_id": safe_tenant},
        )
    summary["rows_updated"] = len(to_update)
    return summary
