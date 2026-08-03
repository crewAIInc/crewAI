"""Tests for the project_id used to link OSS usage to an enterprise account."""

import uuid

import pytest

from crewai_core.project import (
    get_or_create_project_id,
    get_project_id,
    parse_toml,
)


CREW_PYPROJECT = """\
[project]
name = "my_crew"
version = "0.1.0"
dependencies = ["crewai"]

[tool.crewai]
type = "crew"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""


@pytest.fixture
def pyproject(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(CREW_PYPROJECT)
    return path


def test_returns_none_when_no_id_configured(pyproject):
    assert get_project_id(pyproject) is None


def test_mints_and_persists_an_id(pyproject):
    project_id = get_or_create_project_id(pyproject)

    assert uuid.UUID(project_id)
    assert get_project_id(pyproject) == project_id


def test_id_is_stable_across_calls(pyproject):
    first = get_or_create_project_id(pyproject)
    second = get_or_create_project_id(pyproject)

    assert first == second, "must not mint a second id"
    assert uuid.UUID(first)


def test_id_lands_in_the_tool_crewai_table(pyproject):
    project_id = get_or_create_project_id(pyproject)

    data = parse_toml(pyproject.read_text())
    assert data["tool"]["crewai"]["project_id"] == project_id
    assert data["tool"]["crewai"]["type"] == "crew", "existing keys must survive"


def test_other_tables_are_preserved(pyproject):
    get_or_create_project_id(pyproject)

    data = parse_toml(pyproject.read_text())
    assert data["project"]["name"] == "my_crew"
    assert data["project"]["dependencies"] == ["crewai"]
    assert data["build-system"]["build-backend"] == "hatchling.build"


def test_comments_and_formatting_are_preserved(tmp_path):
    """Raw-text editing rather than a TOML round-trip, so comments survive."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '# top comment\n[project]\nname = "x"  # inline comment\n\n[tool.crewai]\ntype = "flow"\n'
    )

    get_or_create_project_id(path)

    content = path.read_text()
    assert "# top comment" in content
    assert "# inline comment" in content


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ('[project]\nname = "x"\n\n[tool.crewai]\ntype = "crew"\n', "table then EOF"),
        ('[tool.crewai]\ntype = "crew"', "no trailing newline"),
        ('[project]\nname = "x"\n', "no tool.crewai table"),
        ('[project]\nname = "x"\n[tool.crewai]\n[other]\na = 1\n', "empty table"),
        (
            '[tool.crewai]\ntype = "crew"\n\n\n[build-system]\nrequires = []\n',
            "blank lines before next table",
        ),
    ],
)
def test_produces_valid_toml_for_varied_layouts(tmp_path, source, label):
    path = tmp_path / "pyproject.toml"
    path.write_text(source)

    project_id = get_or_create_project_id(path)

    assert project_id is not None, label
    data = parse_toml(path.read_text())
    assert data["tool"]["crewai"]["project_id"] == project_id, label


def test_id_does_not_leak_into_a_neighbouring_table(tmp_path):
    """The key must never land under [build-system]."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[tool.crewai]\ntype = "crew"\n\n[build-system]\nrequires = ["hatchling"]\n'
    )

    get_or_create_project_id(path)

    data = parse_toml(path.read_text())
    assert "project_id" in data["tool"]["crewai"]
    assert "project_id" not in data["build-system"]


def test_missing_file_is_not_an_error(tmp_path):
    assert get_or_create_project_id(tmp_path / "nope.toml") is None


def test_malformed_toml_is_not_an_error(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text("this is not [valid toml")

    assert get_project_id(path) is None


def test_read_only_file_is_not_an_error(pyproject):
    """A read-only checkout must not break the command that called this."""
    pyproject.chmod(0o444)
    try:
        project_id = get_or_create_project_id(pyproject)
    finally:
        pyproject.chmod(0o644)

    assert project_id is None


def test_get_project_id_never_creates_anything(pyproject):
    """Library code calls the read-only variant; it must not mutate the file."""
    before = pyproject.read_text()

    assert get_project_id(pyproject) is None

    assert pyproject.read_text() == before


def test_blank_id_is_treated_as_absent(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text('[tool.crewai]\ntype = "crew"\nproject_id = ""\n')

    assert get_project_id(path) is None


def test_ids_are_unique_across_projects(tmp_path):
    ids = set()
    for name in ("a", "b", "c"):
        path = tmp_path / name / "pyproject.toml"
        path.parent.mkdir()
        path.write_text(CREW_PYPROJECT)
        project_id = get_or_create_project_id(path)
        ids.add(project_id)

    assert len(ids) == 3
