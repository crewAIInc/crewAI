"""Tests to ensure documentation links are up-to-date and correct.

These tests verify that outdated or broken links in documentation files
are caught early and prevent regressions (e.g., issue #5253).
"""

import os
import re

import pytest

DOCS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "docs"
)

# The old personal repo URL that was migrated to the crewAIInc org monorepo.
DEPRECATED_TOOLKIT_URL = "github.com/joaomdmoura/crewai-tools"
CORRECT_TOOLKIT_URL = (
    "github.com/crewAIInc/crewAI/tree/main/lib/crewai-tools"
)


def _collect_mdx_files():
    """Collect all .mdx files under the docs directory."""
    mdx_files = []
    for root, _dirs, files in os.walk(DOCS_DIR):
        for f in files:
            if f.endswith(".mdx"):
                mdx_files.append(os.path.join(root, f))
    return mdx_files


@pytest.fixture(scope="module")
def mdx_files():
    files = _collect_mdx_files()
    assert files, f"No .mdx files found under {DOCS_DIR}"
    return files


class TestDocumentationLinks:
    """Ensure documentation links point to the correct repositories."""

    def test_no_deprecated_crewai_tools_link(self, mdx_files):
        """Verify no docs reference the old joaomdmoura/crewai-tools repo.

        The CrewAI tools have moved to the monorepo at
        crewAIInc/crewAI/tree/main/lib/crewai-tools.
        See: https://github.com/crewAIInc/crewAI/issues/5253
        """
        violations = []
        for filepath in mdx_files:
            with open(filepath, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    if DEPRECATED_TOOLKIT_URL in line:
                        rel_path = os.path.relpath(filepath, DOCS_DIR)
                        violations.append(f"  {rel_path}:{line_num}")

        assert not violations, (
            f"Found deprecated CrewAI Toolkit URL ({DEPRECATED_TOOLKIT_URL}) "
            f"in the following doc files. "
            f"Update to: {CORRECT_TOOLKIT_URL}\n"
            + "\n".join(violations)
        )

    def test_crewai_toolkit_links_use_correct_url(self, mdx_files):
        """Verify that CrewAI Toolkit markdown links point to the monorepo."""
        pattern = re.compile(
            r"\[.*?CrewAI.*?Toolkit.*?\]\((https?://[^)]+)\)", re.IGNORECASE
        )
        bad_links = []
        for filepath in mdx_files:
            with open(filepath, "r", encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    for match in pattern.finditer(line):
                        url = match.group(1)
                        if CORRECT_TOOLKIT_URL not in url:
                            rel_path = os.path.relpath(filepath, DOCS_DIR)
                            bad_links.append(
                                f"  {rel_path}:{line_num} -> {url}"
                            )

        assert not bad_links, (
            "Found CrewAI Toolkit links not pointing to the official monorepo "
            f"({CORRECT_TOOLKIT_URL}):\n" + "\n".join(bad_links)
        )
