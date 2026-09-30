#!/usr/bin/env python3
"""Freeze historical doc versions from git tags.

For each release tag listed in ``HISTORICAL_TAGS`` this script extracts the
``docs/en``, ``docs/pt-BR``, ``docs/ko``, ``docs/ar`` directories and the
``docs/enterprise-api.*.yaml`` files at that tag and writes them under
``docs/v<tag>/``. Files that did not yet exist at a given tag are silently
skipped (older tags simply produce smaller snapshots).

Top-level ``docs/v<tag>/`` folders are the Mintlify-idiomatic layout: the
folder name appears verbatim in the URL (``/v1.14.7/en/concepts/agents``),
matching the official versioning examples.

Idempotent: if ``docs/v<tag>/`` already exists the tag is skipped unless
``--force`` is passed.

Usage::

    python scripts/docs/freeze_historical_versions.py
    python scripts/docs/freeze_historical_versions.py --tag 1.14.7
    python scripts/docs/freeze_historical_versions.py --force
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


HISTORICAL_TAGS: list[str] = [
    "1.10.0",
    "1.10.1",
    "1.11.0",
    "1.11.1",
    "1.12.0",
    "1.12.1",
    "1.12.2",
    "1.13.0",
    "1.14.0",
    "1.14.1",
    "1.14.2",
    "1.14.3",
    "1.14.4",
    "1.14.5",
    "1.14.6",
    "1.14.7",
]

SNAPSHOT_PATHS: list[str] = [
    "docs/en",
    "docs/pt-BR",
    "docs/ko",
    "docs/ar",
    "docs/enterprise-api.base.yaml",
    "docs/enterprise-api.en.yaml",
    "docs/enterprise-api.ko.yaml",
    "docs/enterprise-api.pt-BR.yaml",
]


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(out)


def _tag_exists(tag: str) -> bool:
    rc = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}"],
        capture_output=True,
    ).returncode
    return rc == 0


def _paths_present_at_tag(tag: str, paths: list[str]) -> list[str]:
    present: list[str] = []
    for path in paths:
        rc = subprocess.run(
            ["git", "cat-file", "-e", f"{tag}:{path}"],
            capture_output=True,
        ).returncode
        if rc == 0:
            present.append(path)
    return present


def freeze_version(tag: str, *, force: bool = False) -> None:
    root = _repo_root()
    target = root / "docs" / f"v{tag}"

    if target.exists():
        if not force:
            print(f"  skip v{tag} (already frozen at docs/v{tag}/)")
            return
        shutil.rmtree(target)

    if not _tag_exists(tag):
        print(f"  WARN tag {tag} not found, skipping", file=sys.stderr)
        return

    paths = _paths_present_at_tag(tag, SNAPSHOT_PATHS)
    if not paths:
        print(f"  WARN no snapshot paths exist at tag {tag}, skipping", file=sys.stderr)
        return

    target.mkdir(parents=True, exist_ok=True)

    # git archive emits paths verbatim (e.g. docs/en/concepts/agents.mdx).
    # tar --strip-components=1 removes the leading `docs/` segment so the
    # extracted layout under `target` matches `docs/versions/v<tag>/en/...`.
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", tag, *paths],
        cwd=root,
        stdout=subprocess.PIPE,
    )
    untar = subprocess.Popen(
        ["tar", "-x", "--strip-components=1", "-C", str(target)],
        stdin=archive.stdout,
    )
    assert archive.stdout is not None
    archive.stdout.close()
    untar_rc = untar.wait()
    archive_rc = archive.wait()
    if archive_rc != 0 or untar_rc != 0:
        raise RuntimeError(
            f"git archive {tag} failed (archive_rc={archive_rc}, tar_rc={untar_rc})"
        )

    _rewrite_openapi_refs(target, tag)

    file_count = sum(1 for p in target.rglob("*") if p.is_file())
    print(f"  froze v{tag} -> docs/v{tag}/ ({file_count} files)")


# API Reference MDX files reference the OpenAPI spec via an absolute docs-site
# path (e.g. ``openapi: "/enterprise-api.en.yaml GET /foo"``). When a page is
# served from a snapshot we need that path to point at the snapshot's own copy
# of the YAML, otherwise every frozen version would render against the latest
# spec.
_OPENAPI_PATTERN = re.compile(r'(openapi:\s*"\s*)/(enterprise-api\.[^"\s]+\.yaml)')


def _rewrite_openapi_refs(target: Path, tag: str) -> None:
    prefix = f"v{tag}"
    for mdx in target.rglob("*.mdx"):
        text = mdx.read_text(encoding="utf-8")
        new_text, n = _OPENAPI_PATTERN.subn(rf'\1/{prefix}/\2', text)
        if n:
            mdx.write_text(new_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Limit to a specific tag (repeatable). Default: all historical tags.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing snapshot directories.",
    )
    args = parser.parse_args()

    tags = args.tag or HISTORICAL_TAGS
    print(f"Freezing {len(tags)} historical version(s)...")
    for tag in tags:
        freeze_version(tag, force=args.force)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
