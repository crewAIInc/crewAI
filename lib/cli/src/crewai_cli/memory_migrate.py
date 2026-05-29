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
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, TypedDict


_logger = logging.getLogger(__name__)


class MigrateSummary(TypedDict):
    storage_dir: str
    table_name: str
    rows_scanned: int
    rows_to_stamp: int
    rows_with_metadata_key: int
    rows_updated: int


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

    # Scan every row that needs stamping.
    # A row needs stamping if:
    #   - tenant_id is missing/empty, OR
    #   - --from-metadata-key was provided AND metadata[key] differs from row's
    #     current tenant_id (idempotent: re-runs don't re-update unchanged rows).
    rows = table.search().limit(10_000_000).to_list()
    summary["rows_scanned"] = len(rows)

    to_update: list[dict[str, Any]] = []
    for row in rows:
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
