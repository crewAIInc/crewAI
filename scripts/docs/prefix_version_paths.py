#!/usr/bin/env python3
# ruff: noqa: T201
"""Rewrite docs/docs.json to use directory-based versioning.

This script performs the one-time migration that switches every existing
versioned navigation block from referencing the shared ``docs/<lang>/...``
sources to referencing the per-version snapshots under
``docs/v<X.Y.Z>/<lang>/...``. It also inserts a new ``Edge`` entry at the top
of each language's ``versions[]`` array. The Edge entry points at
``docs/edge/<lang>/...`` so unreleased docs live at ``/edge/<lang>/...`` URLs
and never collide with the canonical ``/<lang>/...`` URLs that external links
expect to resolve to the latest released version.

To preserve those canonical URLs, this script also writes a wildcard
``redirects`` block: ``/<lang>/:slug*`` -> ``/<default version>/<lang>/:slug*``.
The release-cut script (``freeze_current_edge.py``) updates the redirect
destination at every release so the canonical URLs always land on the new
default.

After this migration, the version selector behaves honestly: pick v1.10.0 and
you read the v1.10.0 snapshot; pick Edge and you read the current main HEAD;
hit a stale external link and you land on the latest released docs.

Run once::

    python scripts/docs/prefix_version_paths.py

Re-runs are idempotent: pages already starting with ``v<X.Y.Z>/`` (or
``edge/``) are left alone and the Edge entry is only inserted if not already
present.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any


VERSION_SLUG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
LATEST_DEFAULT_VERSION = "v1.14.7"
EDGE_VERSION = "Edge"
EDGE_TAG = "Edge"
EDGE_PREFIX = "edge"
LATEST_TAG = "Latest"

KNOWN_LOCALES = ("en", "pt-BR", "ko", "ar")

# Used by the prune pass to confirm a navigation entry resolves to a real file.
PAGE_EXTENSIONS = (".mdx", ".md")


def _is_version_slug(value: str) -> bool:
    return bool(VERSION_SLUG_RE.match(value))


def _walk_pages(node: Any, transform) -> Any:
    """Recursively walk a navigation subtree, applying ``transform`` to every
    bare page string (i.e. leaves of the ``pages`` lists).
    """
    if isinstance(node, str):
        return transform(node)
    if isinstance(node, list):
        return [_walk_pages(item, transform) for item in node]
    if isinstance(node, dict):
        out = dict(node)
        if "pages" in out:
            out["pages"] = [_walk_pages(p, transform) for p in out["pages"]]
        if "tabs" in out:
            out["tabs"] = [_walk_pages(t, transform) for t in out["tabs"]]
        if "groups" in out:
            out["groups"] = [_walk_pages(g, transform) for g in out["groups"]]
        return out
    return node


def _make_prefixer(locale: str, slug_prefix: str):
    """Return a ``transform`` for ``_walk_pages`` that prefixes pages under
    ``<locale>/`` with ``<slug_prefix>/`` and leaves everything else (e.g.
    ``index``, already-prefixed paths) alone.

    ``slug_prefix`` is the URL-visible segment: ``v1.14.7`` for a frozen
    snapshot, ``edge`` for the rolling channel.
    """
    locale_prefix = f"{locale}/"
    prefix_with_slash = f"{slug_prefix}/"

    def transform(page: str) -> str:
        if page.startswith(prefix_with_slash):
            return page
        if page.startswith(locale_prefix):
            return f"{prefix_with_slash}{page}"
        return page

    return transform


def _prefix_version_entry(entry: dict, locale: str) -> dict:
    """Return a new entry with all page paths under ``locale/`` prefixed with
    ``<version_slug>/`` (no ``versions/`` wrapper, since the slug becomes the
    URL segment). Adds ``tag: "Latest"`` to the default entry.
    """
    version_slug = entry["version"]
    new_entry = _walk_pages(entry, _make_prefixer(locale, version_slug))

    if new_entry.get("default") and "tag" not in new_entry:
        new_entry["tag"] = LATEST_TAG

    return new_entry


def _build_edge_entry(latest_entry: dict, locale: str) -> dict:
    """Clone the current default version's nav structure into an Edge entry
    whose page paths are prefixed with ``edge/<locale>/`` so Edge serves at
    ``/edge/<locale>/...`` URLs and never collides with the canonical
    ``/<locale>/...`` URLs that wildcard redirects own.
    """
    edge = copy.deepcopy(latest_entry)
    edge["version"] = EDGE_VERSION
    edge["tag"] = EDGE_TAG
    edge.pop("default", None)
    # The cloned entry's page paths are still ``<version_slug>/<locale>/...``
    # from the source. Swap the version segment for ``edge``.
    source_prefix = re.compile(rf"^{re.escape(latest_entry['version'])}/")
    locale_prefix = f"{locale}/"
    edge_prefix = f"{EDGE_PREFIX}/"

    def transform(page: str) -> str:
        if page.startswith(edge_prefix):
            return page
        rewritten = source_prefix.sub(edge_prefix, page)
        if rewritten != page:
            return rewritten
        if page.startswith(locale_prefix):
            return f"{edge_prefix}{page}"
        return page

    return _walk_pages(edge, transform)


def _migrate_language_block(block: dict, docs_root: Path) -> dict:
    locale = block["language"]
    versions = block.get("versions", [])
    if not versions:
        return block

    # Detect already-migrated blocks: Edge present and at least one page
    # path starts with ``edge/`` or ``v<digits>.<digits>``.
    already_has_edge = any(v.get("version") == EDGE_VERSION for v in versions)
    looks_prefixed = any(
        isinstance(p, str)
        and (
            p.startswith(f"{EDGE_PREFIX}/") or VERSION_SLUG_RE.match(p.split("/", 1)[0])
        )
        for v in versions
        for p in _flatten_pages(v)
    )
    if already_has_edge and looks_prefixed:
        return block

    latest_entry = next(
        (v for v in versions if v.get("version") == LATEST_DEFAULT_VERSION),
        versions[0],
    )

    # First, prefix every versioned entry so the latest_entry below has the
    # new ``v<X.Y.Z>/<locale>/...`` page paths. We need this BEFORE building
    # Edge because Edge is cloned from the post-prefix latest_entry shape.
    prefixed_entries: list[tuple[dict, dict | None]] = []
    for entry in versions:
        if not _is_version_slug(entry.get("version", "")):
            prefixed_entries.append((entry, entry))
            continue
        prefixed = _prefix_version_entry(entry, locale)
        # The historical docs.json listed pages that did not yet exist at older
        # tags (the old nav-only versioning was lying about which pages were
        # available per release). After prefixing, those paths point at files
        # that don't exist in our frozen snapshots, so we drop them and let
        # empty groups/tabs cascade away.
        pruned = _prune_version_entry(prefixed, docs_root)
        prefixed_entries.append((entry, pruned))

    # Build Edge from the prefixed latest_entry so the clone has consistent
    # shape; we'll rewrite its prefix to ``edge/``.
    latest_prefixed = next(
        (p for orig, p in prefixed_entries if orig is latest_entry and p),
        None,
    )
    if latest_prefixed is None:
        # Latest version has no resolvable pages for this locale; skip Edge.
        new_versions: list[dict] = []
    else:
        edge_entry = _build_edge_entry(latest_prefixed, locale)
        # Verify Edge resolves against docs/edge/<locale>/* on disk.
        edge_pruned = _prune_version_entry(edge_entry, docs_root)
        new_versions = [edge_pruned] if edge_pruned else []

    for _orig, pruned in prefixed_entries:
        if pruned is None:
            continue
        new_versions.append(pruned)

    out = dict(block)
    out["versions"] = new_versions
    return out


def _prune_missing_pages(node: Any, docs_root: Path) -> Any:
    """Remove pages whose target file does not exist under ``docs_root``, and
    cascade-remove now-empty groups/tabs. Returns ``None`` when ``node`` itself
    becomes empty and should be dropped by its parent.

    A "page" is a string leaf inside ``pages``. Strings outside ``pages`` (we
    don't have any in this docs.json today) are preserved.
    """
    if isinstance(node, str):
        for ext in PAGE_EXTENSIONS:
            if (docs_root / f"{node}{ext}").is_file():
                return node
        return None

    if isinstance(node, list):
        pruned = [_prune_missing_pages(item, docs_root) for item in node]
        return [p for p in pruned if p is not None]

    if isinstance(node, dict):
        out: dict = {}
        for key, value in node.items():
            if key in {"pages", "tabs", "groups"}:
                pruned = _prune_missing_pages(value, docs_root)
                if pruned:
                    out[key] = pruned
            else:
                out[key] = value

        if "pages" in node and not out.get("pages"):
            return None
        if "groups" in node and not out.get("groups"):
            return None
        if "tabs" in node and not out.get("tabs"):
            return None
        return out

    return node


def _prune_version_entry(entry: dict, docs_root: Path) -> dict | None:
    """Prune missing pages from a single version entry. Returns ``None`` when
    the entry no longer has any reachable content."""
    pruned = _prune_missing_pages(entry, docs_root)
    if not pruned or not pruned.get("tabs"):
        return None
    return pruned


def _flatten_pages(node: Any) -> list[str]:
    out: list[str] = []

    def visit(n: Any) -> None:
        if isinstance(n, str):
            out.append(n)
        elif isinstance(n, list):
            for x in n:
                visit(x)
        elif isinstance(n, dict):
            for v in n.values():
                visit(v)

    visit(node)
    return out


def _update_redirects(data: dict, default_version: str) -> int:
    """Refresh every redirect so its destination resolves under the default.

    Two passes:

    1. Upsert wildcard ``/<locale>/:slug*`` -> ``/<default>/<locale>/:slug*``
       entries for each known locale so stale canonical URLs keep resolving.
    2. Rewrite the destination of every pre-existing redirect (per-section,
       per-page, redirect-renames, etc.) that currently lands on
       ``/<locale>/...`` so it points at ``/<default>/<locale>/...`` directly.
       Mintlify's link checker doesn't chain redirects, so destinations that
       depend on a second hop count as broken.

    Returns the number of redirect entries inserted or modified.
    """
    redirects = data.setdefault("redirects", [])
    if not isinstance(redirects, list):
        raise RuntimeError("docs.json 'redirects' is not a list")

    upserted = 0
    for locale in KNOWN_LOCALES:
        source = f"/{locale}/:slug*"
        destination = f"/{default_version}/{locale}/:slug*"
        existing = next(
            (r for r in redirects if isinstance(r, dict) and r.get("source") == source),
            None,
        )
        if existing is None:
            redirects.append(
                {"source": source, "destination": destination, "permanent": False}
            )
            upserted += 1
        elif existing.get("destination") != destination:
            existing["destination"] = destination
            existing["permanent"] = False
            upserted += 1

    for entry in redirects:
        if not isinstance(entry, dict):
            continue
        destination = entry.get("destination")
        if not isinstance(destination, str):
            continue
        new_destination = _rewrite_destination_to_version(destination, default_version)
        if new_destination != destination:
            entry["destination"] = new_destination
            upserted += 1

    return upserted


def _rewrite_destination_to_version(destination: str, version_slug: str) -> str:
    """Rewrite a redirect destination to land on ``version_slug`` directly.

    Handles three shapes:

    - ``/<locale>/...``           -> ``/<version_slug>/<locale>/...``
    - ``/v<X.Y.Z>/<locale>/...``  -> ``/<version_slug>/<locale>/...``
    - anything else               -> unchanged
    """
    if not destination.startswith("/"):
        return destination

    parts = destination.lstrip("/").split("/", 2)
    if not parts:
        return destination

    head = parts[0]

    if head in KNOWN_LOCALES:
        return f"/{version_slug}/{destination.lstrip('/')}"

    if VERSION_SLUG_RE.match(head) and len(parts) >= 2 and parts[1] in KNOWN_LOCALES:
        if head == version_slug:
            return destination
        rest = "/".join(parts[1:])
        return f"/{version_slug}/{rest}"

    return destination


def migrate(docs_json: Path) -> tuple[int, int, int, int]:
    data = json.loads(docs_json.read_text(encoding="utf-8"))
    languages = data["navigation"]["languages"]
    docs_root = docs_json.parent

    edge_inserted = 0
    versions_prefixed = 0
    versions_dropped = 0
    for i, block in enumerate(languages):
        before_versions = block.get("versions", [])
        new_block = _migrate_language_block(block, docs_root)
        languages[i] = new_block

        after_versions = new_block.get("versions", [])
        if any(v.get("version") == EDGE_VERSION for v in after_versions) and not any(
            v.get("version") == EDGE_VERSION for v in before_versions
        ):
            edge_inserted += 1
        versions_prefixed += sum(
            1 for v in after_versions if _is_version_slug(v.get("version", ""))
        )
        kept_versioned = sum(
            1 for v in after_versions if _is_version_slug(v.get("version", ""))
        )
        before_versioned = sum(
            1 for v in before_versions if _is_version_slug(v.get("version", ""))
        )
        versions_dropped += before_versioned - kept_versioned

    redirects_upserted = _update_redirects(data, LATEST_DEFAULT_VERSION)

    docs_json.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return edge_inserted, versions_prefixed, versions_dropped, redirects_upserted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-json",
        type=Path,
        default=Path("docs/docs.json"),
        help="Path to docs.json (default: docs/docs.json)",
    )
    args = parser.parse_args()

    if not args.docs_json.exists():
        print(f"ERROR: {args.docs_json} not found", file=sys.stderr)
        return 1

    edge_inserted, versions_prefixed, versions_dropped, redirects_upserted = migrate(
        args.docs_json
    )
    print(
        f"Migrated {args.docs_json}: inserted Edge into {edge_inserted} language "
        f"block(s); rewrote paths in {versions_prefixed} version entries; "
        f"dropped {versions_dropped} (language, version) pairs with no resolvable "
        f"content; upserted {redirects_upserted} canonical-URL redirects."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
