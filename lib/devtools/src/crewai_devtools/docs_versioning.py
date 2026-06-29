"""Freeze the current Edge docs into a per-version snapshot.

Used by ``devtools release`` (and the standalone
``scripts/docs/freeze_current_edge.py`` wrapper) during the docs PR step, which
runs *before* the release tag is created and PyPI publish is triggered. Once
the docs PR merges, the site reflects the new release at ``/v<X.Y.Z>/...`` and
the canonical ``/<lang>/...`` URLs (kept stable for external links) start
redirecting to the new default version.

Layout assumptions (set up by ``scripts/docs/prefix_version_paths.py``):

- ``docs/edge/``                rolling source matching main HEAD
  - ``en/``, ``pt-BR/``, ``ko/``, ``ar/``
  - ``enterprise-api.*.yaml``
- ``docs/v<X.Y.Z>/``            frozen, immutable snapshots
- ``docs/docs.json``            Mintlify config: ``navigation`` + ``redirects``

A freeze does four things:

1. Copy ``docs/edge/*`` into ``docs/v<X.Y.Z>/``.
2. Rewrite ``openapi:`` MDX refs inside the snapshot to point at the snapshot's
   own ``enterprise-api.*.yaml`` (otherwise frozen pages would render against
   whatever YAML happens to be at docs root today).
3. Insert a new version entry into every language's ``versions[]`` block in
   ``docs.json``, place it just after Edge, mark it default + Latest, and demote
   the prior default.
4. Update wildcard redirects in ``docs.json`` so ``/<lang>/:slug*`` lands on the
   new default version.

Idempotent: re-running with a version that already has a snapshot directory
*and* a docs.json entry is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Final


VERSION_RE: Final[re.Pattern[str]] = re.compile(r"^\d+\.\d+\.\d+$")
VERSION_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^v\d+\.\d+\.\d+$")
EDGE_VERSION: Final[str] = "Edge"
EDGE_PREFIX: Final[str] = "edge"
LATEST_TAG: Final[str] = "Latest"

KNOWN_LOCALES: Final[tuple[str, ...]] = ("en", "pt-BR", "ko", "ar")

# Per-snapshot copies are sourced from docs/edge/<name>. The frozen layout
# under docs/v<tag>/ omits the ``edge/`` segment so its URLs are
# ``/v<tag>/<lang>/...``.
SNAPSHOT_PATHS: Final[tuple[str, ...]] = (
    "en",
    "pt-BR",
    "ko",
    "ar",
    "enterprise-api.base.yaml",
    "enterprise-api.en.yaml",
    "enterprise-api.ko.yaml",
    "enterprise-api.pt-BR.yaml",
)

PAGE_EXTENSIONS: Final[tuple[str, ...]] = (".mdx", ".md")

# Matches ``openapi: "/enterprise-api.en.yaml ..."``. The snapshot version of
# the MDX needs the path prefixed with ``v<tag>/`` so the page reads the
# frozen YAML rather than whichever YAML happens to live at docs root.
_OPENAPI_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'(openapi:\s*"\s*)/(enterprise-api\.[^"\s]+\.yaml)'
)


class InvalidVersionError(ValueError):
    """Raised when a freeze is requested with a non-X.Y.Z version string."""


class MissingEdgeSourcesError(RuntimeError):
    """Raised when ``docs/edge/`` is missing or has no snapshot paths."""


@dataclass(frozen=True)
class FreezeResult:
    """Structured outcome of a freeze, for callers that render their own UI."""

    version_slug: str
    snapshot_path: Path
    files_copied: int
    openapi_refs_rewritten: int
    docsjson_entries_inserted: int
    docsjson_entries_skipped: int
    redirects_upserted: int
    snapshot_already_existed: bool


def freeze(version: str, docs_root: Path) -> FreezeResult:
    """Freeze the current Edge into ``docs/v<version>/`` and update docs.json.

    Args:
        version: Release version as ``"X.Y.Z"`` (no leading ``v``).
        docs_root: Path to the ``docs/`` directory.

    Returns:
        ``FreezeResult`` summarising what changed.

    Raises:
        InvalidVersionError: ``version`` is not an X.Y.Z string.
        MissingEdgeSourcesError: ``docs/edge/`` is absent or empty.
    """
    if not VERSION_RE.match(version):
        raise InvalidVersionError(f"{version!r} is not a valid X.Y.Z version string")

    version_slug = f"v{version}"
    target = docs_root / version_slug
    docs_json = docs_root / "docs.json"

    snapshot_already_existed = target.exists()
    if snapshot_already_existed:
        files_copied = 0
        openapi_refs_rewritten = 0
    else:
        files_copied = _copy_snapshot(docs_root, target)
        openapi_refs_rewritten = _rewrite_openapi_refs(target, version_slug)

    inserted, skipped, redirects_upserted = _migrate_docs_json(docs_json, version_slug)

    return FreezeResult(
        version_slug=version_slug,
        snapshot_path=target,
        files_copied=files_copied,
        openapi_refs_rewritten=openapi_refs_rewritten,
        docsjson_entries_inserted=inserted,
        docsjson_entries_skipped=skipped,
        redirects_upserted=redirects_upserted,
        snapshot_already_existed=snapshot_already_existed,
    )


def _copy_snapshot(docs_root: Path, target: Path) -> int:
    """Copy Edge sources under ``docs/edge/`` into ``target``.

    Returns the number of files copied (recursively across directories).
    """
    edge_root = docs_root / EDGE_PREFIX
    if not edge_root.is_dir():
        raise MissingEdgeSourcesError(
            f"Expected Edge sources under {edge_root}/. "
            "Did you forget to migrate Edge into docs/edge/?"
        )

    target.mkdir(parents=True, exist_ok=True)

    count = 0
    for name in SNAPSHOT_PATHS:
        src = edge_root / name
        if not src.exists():
            continue
        dst = target / name
        if src.is_dir():
            shutil.copytree(src, dst)
            count += sum(1 for p in dst.rglob("*") if p.is_file())
        else:
            shutil.copy2(src, dst)
            count += 1

    if count == 0:
        raise MissingEdgeSourcesError(
            f"docs/edge/ exists but contains none of {list(SNAPSHOT_PATHS)}"
        )
    return count


def _rewrite_openapi_refs(target: Path, version_slug: str) -> int:
    """Prefix every ``openapi:`` reference in the snapshot with the version."""
    rewritten = 0
    for mdx in target.rglob("*.mdx"):
        text = mdx.read_text(encoding="utf-8")
        new_text, n = _OPENAPI_PATTERN.subn(rf"\1/{version_slug}/\2", text)
        if n:
            mdx.write_text(new_text, encoding="utf-8")
            rewritten += n
    return rewritten


def _walk_pages(node: Any, transform: Callable[[str], str]) -> Any:
    """Recursively walk a nav subtree, applying ``transform`` to every leaf."""
    if isinstance(node, str):
        return transform(node)
    if isinstance(node, list):
        return [_walk_pages(item, transform) for item in node]
    if isinstance(node, dict):
        out = dict(node)
        for key in ("pages", "tabs", "groups"):
            if key in out:
                out[key] = [_walk_pages(c, transform) for c in out[key]]
        return out
    return node


def _is_version_slug(value: str) -> bool:
    return bool(VERSION_SLUG_RE.match(value))


def _previous_default(versions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the entry currently marked default (or the first versioned)."""
    for v in versions:
        if v.get("default") and _is_version_slug(v.get("version", "")):
            return v
    for v in versions:
        if _is_version_slug(v.get("version", "")):
            return v
    return None


def _build_new_entry(
    previous: dict[str, Any], version_slug: str, locale: str, docs_root: Path
) -> dict[str, Any] | None:
    """Clone the previous default's nav into a new entry for ``version_slug``.

    Page paths are rewritten from ``v<prev>/<locale>/...`` to
    ``v<new>/<locale>/...``. Paths that don't resolve to a file in the
    snapshot are pruned and the now-empty groups/tabs cascade away. Returns
    ``None`` if the locale has no resolvable content under the snapshot (e.g.
    a locale that wasn't present in Edge yet).
    """
    new_entry = copy.deepcopy(previous)
    new_entry["version"] = version_slug
    new_entry["default"] = True
    new_entry["tag"] = LATEST_TAG

    old_prefix = re.compile(rf"^{re.escape(previous['version'])}/")
    locale_prefix = f"{locale}/"
    new_prefix = f"{version_slug}/"

    def transform(page: str) -> str:
        if page.startswith(new_prefix):
            return page
        rewritten = old_prefix.sub(new_prefix, page)
        if rewritten != page:
            return rewritten
        if page.startswith(locale_prefix):
            return f"{new_prefix}{page}"
        return page

    rewritten = _walk_pages(new_entry, transform)
    pruned = _prune_missing_pages(rewritten, docs_root)
    # ``_prune_missing_pages`` recurses across str/list/dict, so its return
    # type is the union of those. We always call it with a dict entry, so we
    # narrow back to ``dict`` here to satisfy the typed signature.
    if not isinstance(pruned, dict) or not pruned.get("tabs"):
        return None
    return pruned


def _prune_missing_pages(node: Any, docs_root: Path) -> Any:
    """Drop pages whose target file is missing; cascade-empty groups/tabs."""
    if isinstance(node, str):
        for ext in PAGE_EXTENSIONS:
            if (docs_root / f"{node}{ext}").is_file():
                return node
        return None

    if isinstance(node, list):
        kept = [_prune_missing_pages(item, docs_root) for item in node]
        return [k for k in kept if k is not None]

    if isinstance(node, dict):
        out: dict[str, Any] = {}
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


def _drop_latest_marker(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    out.pop("default", None)
    if out.get("tag") == LATEST_TAG:
        out.pop("tag")
    return out


def _update_redirects(data: dict[str, Any], version_slug: str) -> int:
    """Make every redirect destination land on the current default version.

    Two passes:

    1. Upsert the wildcard ``/<locale>/:slug*`` -> ``/<version_slug>/<locale>/:slug*``
       entries so stale canonical URLs (``/en/...``, ``/ko/...``, etc.) keep
       resolving.
    2. Rewrite the destination of any pre-existing redirect that lands on a
       bare ``/<locale>/...`` or stale ``/v<old>/<locale>/...`` path so it
       lands on the current default version directly. Mintlify's link checker
       resolves each redirect independently and does not chain through them,
       so a destination that depends on a second redirect counts as broken.

    Returns the number of redirect entries that were inserted or modified.
    """
    redirects = data.setdefault("redirects", [])
    if not isinstance(redirects, list):
        raise RuntimeError("docs.json 'redirects' is not a list")

    upserted = 0

    for locale in KNOWN_LOCALES:
        source = f"/{locale}/:slug*"
        destination = f"/{version_slug}/{locale}/:slug*"
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
        existing_destination = entry.get("destination")
        if not isinstance(existing_destination, str):
            continue
        new_destination = _rewrite_destination_to_version(
            existing_destination, version_slug
        )
        if new_destination != existing_destination:
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


def _migrate_docs_json(docs_json: Path, version_slug: str) -> tuple[int, int, int]:
    """Insert a new versioned entry per language and refresh redirects."""
    data = json.loads(docs_json.read_text(encoding="utf-8"))
    docs_root = docs_json.parent

    inserted = 0
    skipped = 0
    for block in data["navigation"]["languages"]:
        locale = block["language"]
        versions: list[dict[str, Any]] = block.get("versions", [])
        if any(v.get("version") == version_slug for v in versions):
            skipped += 1
            continue

        previous = _previous_default(versions)
        if previous is None:
            skipped += 1
            continue

        new_entry = _build_new_entry(previous, version_slug, locale, docs_root)
        if new_entry is None:
            # Locale has no resolvable content under the snapshot yet (e.g. a
            # locale that didn't exist in Edge). Leave the block untouched.
            skipped += 1
            continue

        updated: list[dict[str, Any]] = []
        for v in versions:
            if v.get("default") or v.get("tag") == LATEST_TAG:
                updated.append(_drop_latest_marker(v))
            else:
                updated.append(v)

        # Insert the new versioned entry just after Edge so the version selector
        # shows: Edge, vNEW (default/Latest), vPREV, vPREV-1, ...
        edge_idx = next(
            (i for i, v in enumerate(updated) if v.get("version") == EDGE_VERSION),
            -1,
        )
        insert_at = edge_idx + 1 if edge_idx >= 0 else 0
        updated.insert(insert_at, new_entry)
        block["versions"] = updated
        inserted += 1

    redirects_upserted = _update_redirects(data, version_slug)

    docs_json.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return inserted, skipped, redirects_upserted
