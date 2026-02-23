"""Regression tests for the repo-local AGENTS.md file.

Validates that AGENTS.md exists at the repository root and contains the
required sections and validation commands so that contributors and
automation agents have deterministic, machine-readable guidance.

Addresses: https://github.com/crewAIInc/crewAI/issues/4564
"""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
AGENTS_MD = REPO_ROOT / "AGENTS.md"


@pytest.fixture()
def agents_md_content() -> str:
    """Read and return the full text of the repo-root AGENTS.md."""
    assert AGENTS_MD.is_file(), (
        f"AGENTS.md not found at repository root ({REPO_ROOT}). "
        "See https://github.com/crewAIInc/crewAI/issues/4564"
    )
    return AGENTS_MD.read_text(encoding="utf-8")


class TestAgentsMdExists:
    """AGENTS.md must be present at the repository root."""

    def test_agents_md_file_exists(self) -> None:
        assert AGENTS_MD.is_file(), (
            f"Expected AGENTS.md at {REPO_ROOT}. "
            "This file is required for contributor and automation-agent guidance."
        )

    def test_agents_md_is_not_empty(self, agents_md_content: str) -> None:
        assert len(agents_md_content.strip()) > 0, "AGENTS.md must not be empty."


class TestAgentsMdRequiredSections:
    """AGENTS.md must contain sections covering validation, safety, and workflow."""

    REQUIRED_HEADINGS = [
        "Repository Layout",
        "Environment Setup",
        "Deterministic Validation Workflow",
        "Safety Boundaries",
        "Commit Convention",
        "Testing Conventions",
        "Code Style",
        "CI Workflows",
    ]

    @pytest.mark.parametrize("heading", REQUIRED_HEADINGS)
    def test_required_section_present(
        self,
        agents_md_content: str,
        heading: str,
    ) -> None:
        assert heading in agents_md_content, (
            f"AGENTS.md is missing required section: '{heading}'"
        )


class TestAgentsMdValidationCommands:
    """AGENTS.md must document the exact validation commands used in CI."""

    REQUIRED_COMMANDS = [
        "uv run ruff check",
        "uv run ruff format",
        "uv run mypy",
        "uv run pytest",
    ]

    @pytest.mark.parametrize("command", REQUIRED_COMMANDS)
    def test_validation_command_documented(
        self,
        agents_md_content: str,
        command: str,
    ) -> None:
        assert command in agents_md_content, (
            f"AGENTS.md must document the validation command: '{command}'"
        )


class TestAgentsMdSafetyRules:
    """AGENTS.md must include key safety rules."""

    SAFETY_KEYWORDS = [
        "Never commit secrets",
        "Never modify VCR cassettes by hand",
        "Never modify tests just to make them pass",
        "--block-network",
    ]

    @pytest.mark.parametrize("keyword", SAFETY_KEYWORDS)
    def test_safety_rule_present(
        self,
        agents_md_content: str,
        keyword: str,
    ) -> None:
        assert keyword in agents_md_content, (
            f"AGENTS.md is missing safety rule containing: '{keyword}'"
        )


class TestAgentsMdWorkspaceStructure:
    """AGENTS.md must accurately describe the workspace layout."""

    WORKSPACE_MEMBERS = [
        "lib/crewai/",
        "lib/crewai-tools/",
        "lib/crewai-files/",
        "lib/devtools/",
    ]

    @pytest.mark.parametrize("member", WORKSPACE_MEMBERS)
    def test_workspace_member_documented(
        self,
        agents_md_content: str,
        member: str,
    ) -> None:
        # Normalize: the file may reference paths with or without trailing /
        member_stem = member.rstrip("/")
        assert member_stem in agents_md_content, (
            f"AGENTS.md must document workspace member: '{member_stem}'"
        )
