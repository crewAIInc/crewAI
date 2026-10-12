#!/usr/bin/env python3
# ruff: noqa: T201
"""Refresh the checked-in LiteLLM model catalog.

Downloads LiteLLM's ``model_prices_and_context_window.json`` with curl and
replaces the snapshot used by context-window lookup:

``lib/crewai/src/crewai/llms/litellm_catalog/model_prices_and_context_window.json``

The URL matches ``crewai.constants.JSON_URL``.

Usage::

    python scripts/update_litellm_catalog.py
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


CATALOG_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
CATALOG_RELATIVE = Path(
    "lib/crewai/src/crewai/llms/litellm_catalog/model_prices_and_context_window.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def update_catalog(repo_root: Path | None = None) -> Path:
    """Download the LiteLLM catalog with curl and write it into the checkout.

    Args:
        repo_root: Checkout root. Defaults to the parent of ``scripts/``.

    Returns:
        Path of the catalog file that was written.

    Raises:
        RuntimeError: curl is missing or the download fails.
        ValueError: the response is not a LiteLLM catalog object.
        json.JSONDecodeError: the response is not valid JSON.
    """
    root = _repo_root() if repo_root is None else repo_root
    destination = root / CATALOG_RELATIVE
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not installed or not on PATH")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=".model_prices_and_context_window.",
        suffix=".json",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)

    try:
        print(f"Downloading {CATALOG_URL}")
        completed = subprocess.run(  # noqa: S603 - fixed curl binary and catalog URL
            [
                curl,
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--output",
                str(temp_path),
                CATALOG_URL,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"curl exited {completed.returncode}"
            raise RuntimeError(detail)

        payload = json.loads(temp_path.read_bytes())
        if not isinstance(payload, dict) or "sample_spec" not in payload:
            raise ValueError("Downloaded catalog is not a LiteLLM model prices object")

        print(f"Updating {CATALOG_RELATIVE}")
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return destination


def main() -> int:
    """Download the catalog and print the path that was updated."""
    try:
        path = update_catalog()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Updated {path.relative_to(_repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
