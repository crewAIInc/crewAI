from pathlib import Path

import pytest

from crewai_tools.tools.directory_read_tool.directory_read_tool import (
    DirectoryReadTool,
)


def test_lists_actual_path_when_descendant_repeats_base_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "workspace"
    repeated = base / base.relative_to(base.anchor)
    repeated.mkdir(parents=True)
    target = repeated / "sentinel.txt"
    target.write_text("content", encoding="utf-8")

    result = DirectoryReadTool()._run(directory=str(base))

    assert result == f"File paths: \n-{target}"
    assert target.exists()


def _outside_dir(tmp_path: Path, monkeypatch) -> Path:
    """Run from tmp_path/workspace and return a directory outside it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "q3.md").write_text("Q3", encoding="utf-8")
    return reports


def test_constructor_directory_outside_working_directory_is_listed(
    tmp_path: Path, monkeypatch
) -> None:
    """A developer-declared directory is trusted even outside the sandbox."""
    reports = _outside_dir(tmp_path, monkeypatch)

    tool = DirectoryReadTool(directory=str(reports))

    assert tool.run() == f"File paths: \n-{reports / 'q3.md'}"


def test_declared_relative_directory_survives_chdir(
    tmp_path: Path, monkeypatch
) -> None:
    """The declared directory is pinned at construction, not per call."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "original.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "sub"
    (nested / "data").mkdir(parents=True)
    (nested / "data" / "other.txt").write_text("x", encoding="utf-8")

    tool = DirectoryReadTool(directory="data")
    monkeypatch.chdir(nested)

    assert tool.run() == f"File paths: \n-{tmp_path / 'data' / 'original.txt'}"


def test_declared_directory_that_is_not_a_directory_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    reports = _outside_dir(tmp_path, monkeypatch)

    tool = DirectoryReadTool(directory=str(reports / "q3.md"))

    with pytest.raises(ValueError, match="is not a directory"):
        tool._run()


def test_runtime_directory_outside_working_directory_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    """Directories chosen at runtime still go through the containment check."""
    reports = _outside_dir(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="outside the allowed directory") as exc:
        DirectoryReadTool()._run(directory=str(reports))

    # Point users at base_dir, not the process-wide escape hatch.
    assert "base_dir" in str(exc.value)
    assert "CREWAI_TOOLS_ALLOW_UNSAFE_PATHS" not in str(exc.value)


def test_base_dir_widens_the_sandbox_for_runtime_directories(
    tmp_path: Path, monkeypatch
) -> None:
    reports = _outside_dir(tmp_path, monkeypatch)
    elsewhere = tmp_path.parent

    tool = DirectoryReadTool(base_dir=str(tmp_path))

    assert tool._run(directory=str(reports)) == f"File paths: \n-{reports / 'q3.md'}"
    with pytest.raises(ValueError, match="outside the allowed directory"):
        tool._run(directory=str(elsewhere))


def test_relative_base_dir_is_anchored_at_construction(
    tmp_path: Path, monkeypatch
) -> None:
    """A relative base_dir must not follow a later chdir."""
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / "allowed"
    (allowed / "data").mkdir(parents=True)
    (allowed / "data" / "file.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()

    tool = DirectoryReadTool(base_dir="allowed")
    assert tool.base_dir == str(allowed)

    monkeypatch.chdir(nested)
    assert tool._run(directory="data") == (
        f"File paths: \n-{allowed / 'data' / 'file.txt'}"
    )


def test_declared_directory_survives_a_serialization_round_trip(
    tmp_path: Path, monkeypatch
) -> None:
    """model_dump drops private attrs, so the pin must be rebuilt on restore."""
    reports = _outside_dir(tmp_path, monkeypatch)

    tool = DirectoryReadTool(directory=str(reports))
    restored = DirectoryReadTool.model_validate(tool.model_dump())

    assert restored._declared_realpath == tool._declared_realpath
    assert restored.run() == f"File paths: \n-{reports / 'q3.md'}"
