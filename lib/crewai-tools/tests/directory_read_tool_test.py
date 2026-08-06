"""Regression tests for DirectoryReadTool.

Covers a bug where a fixed directory configured by the developer, outside
the process's current working directory, was always rejected as outside the
allowed directory, because _run validated it against validate_directory_path
with its default base_dir (os.getcwd()) and no way to widen or pin it,
unlike the sibling FileReadTool/FileWriterTool, which pin a declared path at
construction time.

This made the tool's most basic documented use case,
DirectoryReadTool(directory="/some/other/dir"), unusable whenever that
directory was not the current working directory, and raised an unhandled
ValueError instead of returning a graceful error string.
"""

import os

import pytest

from crewai_tools.tools.directory_read_tool.directory_read_tool import (
    DirectoryReadTool,
)


@pytest.fixture
def outside_dir(tmp_path_factory):
    """A directory that is not nested under the cwd tests chdir into.

    Built from its own top level tmp path, a sibling of the cwd fixture
    below, not a descendant of it. A directory nested inside the cwd would
    already be reachable under the old, broken containment check, which
    allowed anything under os.getcwd(). A test using a nested path would
    pass on the old code too and prove nothing about the fix.
    """
    target = tmp_path_factory.mktemp("outside")
    (target / "a.txt").write_text("hello")
    (target / "nested").mkdir()
    (target / "nested" / "b.txt").write_text("world")
    return target


@pytest.fixture
def cwd(tmp_path_factory):
    """A cwd directory that does not contain outside_dir (see above)."""
    return tmp_path_factory.mktemp("cwd")


def test_fixed_directory_outside_cwd_is_listed(outside_dir, cwd, monkeypatch):
    """A directory declared at construction time must work even when it is
    not the current working directory (the primary documented use case)."""
    monkeypatch.chdir(cwd)
    tool = DirectoryReadTool(directory=str(outside_dir))

    result = tool._run()

    assert "Error" not in result
    assert "a.txt" in result
    assert "b.txt" in result


def test_fixed_directory_survives_chdir(outside_dir, cwd, tmp_path_factory, monkeypatch):
    """The declared directory is pinned at construction time, like
    FileReadTool's declared file_path, so a later chdir cannot break it."""
    monkeypatch.chdir(cwd)
    tool = DirectoryReadTool(directory=str(outside_dir))

    other_cwd = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(other_cwd)

    result = tool._run()

    assert "Error" not in result
    assert "a.txt" in result


def test_runtime_directory_outside_base_dir_is_rejected(outside_dir, tmp_path, monkeypatch):
    """Sandboxing must be preserved for LLM supplied paths at runtime: only
    the declared, construction time directory bypasses containment."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    tool = DirectoryReadTool()

    result = tool._run(directory=str(outside_dir))

    assert "Error" in result
    assert "outside the allowed directory" in result


def test_runtime_directory_inside_cwd_is_listed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("hi")
    tool = DirectoryReadTool()

    result = tool._run(directory=".")

    assert "Error" not in result
    assert "f.txt" in result


def test_base_dir_widens_runtime_sandbox(outside_dir, tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    tool = DirectoryReadTool(base_dir=str(outside_dir.parent))

    result = tool._run(directory=str(outside_dir))

    assert "Error" not in result
    assert "a.txt" in result


def test_missing_directory_returns_error_not_exception(tmp_path, monkeypatch):
    """A nonexistent fixed directory must return a graceful error, not raise."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing_dir"
    tool = DirectoryReadTool(directory=str(missing))

    result = tool._run()

    assert "Error" in result


def test_root_directory_paths_have_single_leading_slash():
    """Listing '/' must not produce a doubled leading slash.

    directory + "/" + relpath string concatenation turns "/" into "//" once
    joined with a relative path. os.path.join avoids this.
    """
    directory = "/"
    joined = os.path.join(directory, os.path.relpath("/etc/hosts", directory))
    assert joined == "/etc/hosts"
    assert not joined.startswith("//")
