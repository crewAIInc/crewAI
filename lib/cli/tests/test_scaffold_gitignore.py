"""Every runnable scaffold ignores `.crewai/`, where crewAI records the last run."""

from __future__ import annotations

from pathlib import Path

import pytest


TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "crewai_cli" / "templates"


@pytest.mark.parametrize("template", ["crew", "flow", "json_crew", "declarative_flow"])
def test_the_scaffold_gitignore_covers_the_crewai_directory(template):
    lines = (TEMPLATES / template / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".crewai/" in lines
    assert ".env" in lines
