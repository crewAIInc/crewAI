from __future__ import annotations

from pathlib import Path

from crewai_core.project import get_project_description, get_project_name


def test_project_attributes_read_a_non_ascii_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "research_crew"\n'
        'description = "数据分析团队 · Café"\n'
        'dependencies = ["crewai[tools]"]\n',
        encoding="utf-8",
    )

    assert get_project_name(str(pyproject), require=True) == "research_crew"
    assert get_project_description(str(pyproject)) == "数据分析团队 · Café"
