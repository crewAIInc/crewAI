"""FilePath must reject traversal without rejecting dots in a filename."""

from pathlib import Path

import pytest

from crewai_files import FilePath


def test_dots_inside_filename_are_allowed(tmp_path: Path):
    path = tmp_path / "data..2024.csv"
    path.write_text("value\n1\n", encoding="utf-8")

    assert FilePath(path=path).path.name == "data..2024.csv"


def test_parent_directory_component_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Path traversal not allowed"):
        FilePath(path=tmp_path / ".." / "secret.csv")
