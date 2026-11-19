"""Tests that FilePath cannot read files outside the working directory."""

from pathlib import Path

from crewai_files.core.sources import FilePath
import pytest


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return project


def test_regular_file_inside_cwd_is_allowed(project: Path) -> None:
    (project / "ok.txt").write_text("ok")

    assert FilePath(path=Path("ok.txt")).filename == "ok.txt"


def test_symlink_inside_cwd_is_allowed(project: Path) -> None:
    (project / "real.txt").write_text("ok")
    (project / "link.txt").symlink_to(project / "real.txt")

    assert FilePath(path=Path("link.txt")).filename == "link.txt"


def test_file_symlink_outside_cwd_is_rejected(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (project / "link.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes allowed directory"):
        FilePath(path=Path("link.txt"))


def test_parent_directory_symlink_outside_cwd_is_rejected(
    project: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (project / "linkdir").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes allowed directory"):
        FilePath(path=Path("linkdir/secret.txt"))


def test_symlink_to_sibling_dir_sharing_cwd_prefix_is_rejected(
    project: Path, tmp_path: Path
) -> None:
    sibling = tmp_path / "project-secrets"
    sibling.mkdir()
    (sibling / "keys.txt").write_text("secret")
    (project / "keys_link.txt").symlink_to(sibling / "keys.txt")

    with pytest.raises(ValueError, match="escapes allowed directory"):
        FilePath(path=Path("keys_link.txt"))
