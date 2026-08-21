"""Keep README Getting Started examples aligned with JSON-first crew files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from crewai.project.json_loader import strip_jsonc_comments, validate_crew_project

REPO_ROOT = Path(__file__).resolve().parents[4]
README_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "lib" / "crewai" / "README.md",
)


def _getting_started_section(markdown: str) -> str:
    match = re.search(
        r"## Getting Started\n(.*?)(?:\n## Key Features\n)",
        markdown,
        flags=re.DOTALL,
    )
    assert match is not None, "Getting Started section is missing"
    return match.group(1)


def _extract_fenced_blocks(markdown: str, language: str) -> list[str]:
    pattern = rf"```{language}\n(.*?)```"
    return re.findall(pattern, markdown, flags=re.DOTALL)


@pytest.mark.parametrize("readme_path", README_PATHS, ids=["root", "package"])
def test_readme_getting_started_documents_json_first_and_classic(
    readme_path: Path,
) -> None:
    section = _getting_started_section(readme_path.read_text(encoding="utf-8"))

    assert "crewai create crew <project_name>" in section
    assert "agents/*.jsonc" in section
    assert "crew.jsonc" in section
    assert "--classic" in section
    assert "config/agents.yaml" in section
    assert "config/tasks.yaml" in section
    assert "crew.py" in section
    assert "python src/latest_ai_development/main.py" in section


@pytest.mark.parametrize("readme_path", README_PATHS, ids=["root", "package"])
def test_readme_latest_ai_development_example_matches_generated_layout(
    readme_path: Path,
    tmp_path: Path,
) -> None:
    section = _getting_started_section(readme_path.read_text(encoding="utf-8"))
    blocks = _extract_fenced_blocks(section, "jsonc")
    assert len(blocks) == 3

    researcher, reporting_analyst, crew = (
        json.loads(strip_jsonc_comments(block)) for block in blocks
    )

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "researcher.jsonc").write_text(blocks[0], encoding="utf-8")
    (agents_dir / "reporting_analyst.jsonc").write_text(blocks[1], encoding="utf-8")
    crew_path = tmp_path / "crew.jsonc"
    crew_path.write_text(blocks[2], encoding="utf-8")

    project = validate_crew_project(crew_path)

    assert project.agent_names == ["researcher", "reporting_analyst"]
    assert researcher["role"] == "{topic} Senior Data Researcher"
    assert researcher["tools"] == ["SerperDevTool"]
    assert reporting_analyst["role"] == "{topic} Reporting Analyst"
    assert crew["name"] == "latest-ai-development"
    assert crew["process"] == "sequential"
    assert crew["inputs"] == {"topic": "AI Agents", "current_year": "2026"}
    assert [task["name"] for task in project.task_definitions] == [
        "research_task",
        "reporting_task",
    ]
    assert project.task_definitions[1]["context"] == ["research_task"]
    assert project.task_definitions[1]["output_file"] == "report.md"
    assert project.task_definitions[1]["agent"] == "reporting_analyst"
