#!/usr/bin/env python3
# ruff: noqa: T201, S607
"""Standalone CLI wrapper around :mod:`crewai_devtools.docs_versioning`.

``devtools release`` calls the same freeze logic during its docs PR step; this
script is the manual escape hatch for one-off freezes (e.g. retroactively
freezing a forgotten release, or freezing without going through the full
release flow).

Usage::

    python scripts/docs/freeze_current_edge.py 1.15.0

Idempotent: re-running with the same version is a no-op (existing snapshot
directory and existing docs.json entry are both detected).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from crewai_devtools.docs_versioning import (
    InvalidVersionError,
    MissingEdgeSourcesError,
    freeze,
)


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        help='New release version as "X.Y.Z" (no leading v). Example: 1.15.0',
    )
    args = parser.parse_args()

    docs_root = _repo_root() / "docs"
    try:
        result = freeze(args.version, docs_root)
    except InvalidVersionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except MissingEdgeSourcesError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    relative_snapshot = result.snapshot_path.relative_to(docs_root.parent)
    if result.snapshot_already_existed:
        print(f"Snapshot directory already exists: {relative_snapshot}")
        print("Skipping copy. Re-running docs.json migration only.")
    else:
        print(
            f"Froze Edge -> {relative_snapshot} "
            f"({result.files_copied} files, "
            f"{result.openapi_refs_rewritten} openapi refs rewritten)."
        )

    print(
        f"Updated docs/docs.json: inserted {result.version_slug} into "
        f"{result.docsjson_entries_inserted} language block(s), "
        f"skipped {result.docsjson_entries_skipped}, "
        f"upserted {result.redirects_upserted} canonical-URL redirects."
    )
    print()
    print("Commit message suggestion:")
    print(f"  [docs-freeze] snapshot docs for {result.version_slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
