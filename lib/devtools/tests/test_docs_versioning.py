"""Tests for the Edge -> snapshot freeze used by the docs PR step."""

from __future__ import annotations

import json
from pathlib import Path

from crewai_devtools.docs_versioning import (
    InvalidVersionError,
    MissingEdgeSourcesError,
    freeze,
)
import pytest


def _build_docs_root(tmp_path: Path) -> Path:
    """Build a minimal docs/ tree with one previous snapshot + Edge + docs.json.

    The shape mirrors the real repo: an Edge directory with per-locale folders
    and YAMLs, one previously-frozen v1.14.7 snapshot, and a docs.json with
    Edge + the previous default in the version selector plus canonical-URL
    wildcard redirects.
    """
    docs = tmp_path / "docs"

    # Edge sources (what the release will freeze).
    edge_en = docs / "edge" / "en"
    edge_en.mkdir(parents=True)
    (edge_en / "introduction.mdx").write_text("# Intro (Edge)\n")
    (edge_en / "changelog.mdx").write_text("---\ntitle: Changelog\n---\n")
    (edge_en / "api.mdx").write_text(
        '---\nopenapi: "/enterprise-api.en.yaml GET /foo"\n---\n'
    )
    # A page added to Edge after the previous release. It exists as a file and
    # is wired into the Edge nav, but is intentionally absent from the v1.14.7
    # nav below — the freeze must still surface it in the new version.
    (edge_en / "datadog.mdx").write_text("# Datadog (Edge)\n")
    (docs / "edge" / "enterprise-api.en.yaml").write_text("openapi: 3.0.0\n")

    # A pre-existing frozen snapshot to clone the nav structure from.
    snap_en = docs / "v1.14.7" / "en"
    snap_en.mkdir(parents=True)
    (snap_en / "introduction.mdx").write_text("# Intro (1.14.7)\n")
    (snap_en / "changelog.mdx").write_text("---\ntitle: Changelog\n---\n")
    (snap_en / "api.mdx").write_text(
        '---\nopenapi: "/v1.14.7/enterprise-api.en.yaml GET /foo"\n---\n'
    )

    docs_json = {
        "navigation": {
            "languages": [
                {
                    "language": "en",
                    "versions": [
                        {
                            "version": "Edge",
                            "tag": "Edge",
                            "tabs": [
                                {
                                    "tab": "Guides",
                                    "pages": [
                                        "edge/en/introduction",
                                        "edge/en/changelog",
                                        "edge/en/api",
                                        "edge/en/datadog",
                                    ],
                                }
                            ],
                        },
                        {
                            "version": "v1.14.7",
                            "default": True,
                            "tag": "Latest",
                            "tabs": [
                                {
                                    "tab": "Guides",
                                    "pages": [
                                        "v1.14.7/en/introduction",
                                        "v1.14.7/en/changelog",
                                        "v1.14.7/en/api",
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ]
        },
        "redirects": [
            {
                "source": "/en/:slug*",
                "destination": "/v1.14.7/en/:slug*",
                "permanent": False,
            }
        ],
    }
    (docs / "docs.json").write_text(json.dumps(docs_json, indent=2) + "\n")
    return docs


class TestFreeze:
    def test_copies_edge_files_into_snapshot(self, tmp_path: Path) -> None:
        docs = _build_docs_root(tmp_path)

        result = freeze("1.15.0", docs)

        snapshot = docs / "v1.15.0"
        assert snapshot.is_dir()
        assert (snapshot / "en" / "introduction.mdx").read_text() == "# Intro (Edge)\n"
        assert (snapshot / "enterprise-api.en.yaml").read_text() == "openapi: 3.0.0\n"
        assert result.snapshot_path == snapshot
        assert result.files_copied >= 4
        assert result.snapshot_already_existed is False

    def test_rewrites_openapi_refs_to_snapshot_yaml(self, tmp_path: Path) -> None:
        docs = _build_docs_root(tmp_path)

        result = freeze("1.15.0", docs)

        frozen_api = (docs / "v1.15.0" / "en" / "api.mdx").read_text()
        assert "/v1.15.0/enterprise-api.en.yaml" in frozen_api
        # The Edge source must NOT be rewritten — it stays generic so the next
        # release freezes pick up the same edit.
        edge_api = (docs / "edge" / "en" / "api.mdx").read_text()
        assert "/v1.15.0/" not in edge_api
        assert "/enterprise-api.en.yaml" in edge_api
        assert result.openapi_refs_rewritten >= 1

    def test_inserts_version_after_edge_and_demotes_previous_default(
        self, tmp_path: Path
    ) -> None:
        docs = _build_docs_root(tmp_path)

        freeze("1.15.0", docs)

        data = json.loads((docs / "docs.json").read_text())
        versions = data["navigation"]["languages"][0]["versions"]
        labels = [v["version"] for v in versions]
        assert labels == ["Edge", "v1.15.0", "v1.14.7"]

        new_entry = versions[1]
        assert new_entry["default"] is True
        assert new_entry["tag"] == "Latest"
        # Page paths in the new entry must point at the new snapshot.
        page_strs = [
            p for tab in new_entry["tabs"] for p in tab["pages"] if isinstance(p, str)
        ]
        assert all(p.startswith("v1.15.0/en/") for p in page_strs)

        previous = versions[2]
        assert "default" not in previous
        assert previous.get("tag") != "Latest"

    def test_new_version_nav_is_cloned_from_edge_not_previous(
        self, tmp_path: Path
    ) -> None:
        # Regression: the new version's nav must come from Edge so pages added
        # to Edge since the last release ship in the freeze. Cloning the
        # previous version's nav silently dropped them (the file was copied
        # into the snapshot but never linked in the version selector).
        docs = _build_docs_root(tmp_path)

        freeze("1.15.0", docs)

        data = json.loads((docs / "docs.json").read_text())
        versions = data["navigation"]["languages"][0]["versions"]
        new_entry = next(v for v in versions if v["version"] == "v1.15.0")
        pages = [p for tab in new_entry["tabs"] for p in tab["pages"]]
        assert "v1.15.0/en/datadog" in pages
        # And the file is present in the snapshot it points at.
        assert (docs / "v1.15.0" / "en" / "datadog.mdx").is_file()

    def test_updates_canonical_url_redirect_to_new_default(
        self, tmp_path: Path
    ) -> None:
        docs = _build_docs_root(tmp_path)

        result = freeze("1.15.0", docs)

        data = json.loads((docs / "docs.json").read_text())
        en_redirect = next(r for r in data["redirects"] if r["source"] == "/en/:slug*")
        assert en_redirect["destination"] == "/v1.15.0/en/:slug*"
        assert en_redirect["permanent"] is False
        assert result.redirects_upserted >= 1

    def test_rewrites_stale_per_section_redirects_to_new_default(
        self, tmp_path: Path
    ) -> None:
        # docs.json carries pre-existing per-section/per-page redirects whose
        # destinations point at /<locale>/... or /v<prev>/<locale>/...; those
        # need to land on the current default version directly because
        # Mintlify's link-checker doesn't follow redirect chains.
        docs = _build_docs_root(tmp_path)
        data = json.loads((docs / "docs.json").read_text())
        data["redirects"].extend(
            [
                {"source": "/concepts/:path*", "destination": "/en/concepts/:path*"},
                {
                    "source": "/api-reference/:path*",
                    "destination": "/v1.14.7/en/api-reference/:path*",
                },
                {"source": "/introduction", "destination": "/en/introduction"},
                {"source": "/external", "destination": "https://example.com/"},
            ]
        )
        (docs / "docs.json").write_text(json.dumps(data, indent=2) + "\n")

        freeze("1.15.0", docs)

        data = json.loads((docs / "docs.json").read_text())
        by_source = {r["source"]: r["destination"] for r in data["redirects"]}
        assert by_source["/concepts/:path*"] == "/v1.15.0/en/concepts/:path*"
        assert by_source["/api-reference/:path*"] == "/v1.15.0/en/api-reference/:path*"
        assert by_source["/introduction"] == "/v1.15.0/en/introduction"
        # Absolute URLs and other non-locale destinations are left alone.
        assert by_source["/external"] == "https://example.com/"

    def test_is_idempotent_when_snapshot_already_exists(self, tmp_path: Path) -> None:
        docs = _build_docs_root(tmp_path)

        freeze("1.15.0", docs)
        before = (docs / "docs.json").read_text()
        result = freeze("1.15.0", docs)

        assert result.snapshot_already_existed is True
        assert result.files_copied == 0
        assert result.openapi_refs_rewritten == 0
        # docs.json shape doesn't change on the second run because the version
        # is already registered.
        assert result.docsjson_entries_inserted == 0
        assert (docs / "docs.json").read_text() == before

    def test_rejects_invalid_version_string(self, tmp_path: Path) -> None:
        docs = _build_docs_root(tmp_path)

        with pytest.raises(InvalidVersionError):
            freeze("v1.15.0", docs)
        with pytest.raises(InvalidVersionError):
            freeze("1.15", docs)
        with pytest.raises(InvalidVersionError):
            freeze("1.15.0a1", docs)

    def test_rejects_missing_edge_directory(self, tmp_path: Path) -> None:
        docs = _build_docs_root(tmp_path)
        import shutil

        shutil.rmtree(docs / "edge")

        with pytest.raises(MissingEdgeSourcesError):
            freeze("1.15.0", docs)
