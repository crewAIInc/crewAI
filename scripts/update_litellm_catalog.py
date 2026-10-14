#!/usr/bin/env python3
# ruff: noqa: T201
"""Refresh the checked-in LiteLLM model catalog.

Manual wrapper around :func:`crewai_devtools.litellm_catalog_update.update_catalog`.
``devtools release`` calls that function after the version-bump PR merges.

Usage::

    python scripts/update_litellm_catalog.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from crewai_devtools.litellm_catalog_update import update_catalog


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    """Download the catalog and print the path that was updated."""
    repo_root = _repo_root()
    try:
        path = update_catalog(repo_root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Updated {path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
