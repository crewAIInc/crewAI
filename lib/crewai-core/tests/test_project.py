from __future__ import annotations

from pathlib import Path

import pytest

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


def test_missing_required_attribute_exits_with_status_1(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "research_crew"\n'
        'dependencies = ["crewai"]\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        get_project_description(str(pyproject), require=True)

    assert exc_info.value.code == 1


def test_missing_required_pyproject_exits_with_status_1(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        get_project_name(str(tmp_path / "pyproject.toml"), require=True)

    assert exc_info.value.code == 1
