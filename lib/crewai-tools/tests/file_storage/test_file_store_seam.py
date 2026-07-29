"""The file tools must route all I/O through the registered store.

These tests stand in a store that keeps everything in a dict, with no
filesystem behind it at all. If a tool reaches past the seam to `open()` or
`os.path` the assertions fail, which is the point: it is the guarantee any
non-filesystem store depends on.
"""

from __future__ import annotations

import io
import posixpath

from crewai_tools import FileReadTool, FileWriterTool
from crewai_tools.file_storage import (
    FileStore,
    register_file_store_factory,
    reset_file_store_factory,
    resolve_file_store,
)
from crewai_tools.file_storage.local import LocalFileStore
import pytest


class MemoryFileStore:
    """A store with no filesystem: paths are keys in a dict.

    Modelled on a remote object store — POSIX-ish paths, a flat namespace, no
    symlinks, and containment by prefix rather than by `realpath`.
    """

    label = "memory"

    def __init__(self, root: str = "/ws") -> None:
        self.root = root
        self.files: dict[str, str] = {}
        self.dirs: set[str] = {root}

    def _abs(self, path: str, base_dir: str | None = None) -> str:
        base = base_dir or self.root
        joined = path if path.startswith("/") else posixpath.join(base, path)
        return posixpath.normpath(joined)

    def normalize(self, path: str, base_dir: str | None = None) -> str:
        return self._abs(path, base_dir)

    def resolve(self, path: str, base_dir: str | None = None) -> str:
        resolved = self._abs(path, base_dir)
        # Confine to base_dir when one is given, else to the store root. A
        # remote store has to honour base_dir the same way the local one does,
        # or the tools' sandbox argument would silently mean nothing.
        root = posixpath.normpath(base_dir) if base_dir else self.root
        if resolved != root and not resolved.startswith(root.rstrip("/") + "/"):
            raise ValueError(
                f"Path '{posixpath.basename(resolved)}' is outside the allowed "
                f"directory."
            )
        return resolved

    def resolve_within(self, directory: str, filename: str) -> str:
        resolved = self._abs(filename, directory)
        if resolved == directory or not resolved.startswith(directory + "/"):
            raise ValueError("the filename must not escape the target directory")
        return resolved

    def display(self, resolved: str, base: str | None = None) -> str:
        base = base or self.root
        if resolved.startswith(base + "/"):
            return resolved[len(base) + 1 :]
        return posixpath.basename(resolved)

    def exists(self, resolved: str) -> bool:
        return resolved in self.files

    def ensure_parent(self, resolved: str) -> None:
        parent = posixpath.dirname(resolved)
        if parent in self.files:
            raise FileExistsError(parent)
        self.dirs.add(parent)

    def open_text(self, resolved: str, encoding: str):
        if resolved not in self.files:
            raise FileNotFoundError(resolved)
        return io.StringIO(self.files[resolved])

    def write_text(
        self, resolved: str, content: str, encoding: str, *, overwrite: bool
    ) -> None:
        if resolved in self.files and not overwrite:
            raise FileExistsError(resolved)
        self.files[resolved] = content


@pytest.fixture
def store():
    memory = MemoryFileStore()
    register_file_store_factory(lambda: memory)
    yield memory
    reset_file_store_factory()


def test_default_store_is_local():
    assert isinstance(resolve_file_store(), LocalFileStore)


def test_memory_store_satisfies_the_protocol(store):
    assert isinstance(store, FileStore)


def test_writer_writes_through_the_store(store, tmp_path, monkeypatch):
    # cwd is a real, empty directory: nothing may touch it.
    monkeypatch.chdir(tmp_path)

    result = FileWriterTool()._run(
        filename="report.md", content="# Report", overwrite=True
    )

    assert "successfully written" in result
    assert store.files == {"/ws/report.md": "# Report"}
    assert list(tmp_path.iterdir()) == []


def test_reader_reads_through_the_store(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store.files["/ws/notes.txt"] = "line 1\nline 2\nline 3\n"

    assert FileReadTool()._run(file_path="notes.txt") == "line 1\nline 2\nline 3\n"


def test_round_trip_between_the_two_tools(store, tmp_path, monkeypatch):
    """The reader must see what the writer wrote — the asymmetry that makes
    an ephemeral runtime unusable."""
    monkeypatch.chdir(tmp_path)

    FileWriterTool()._run(filename="out/data.csv", content="a,b\n1,2\n", overwrite=True)
    read_back = FileReadTool()._run(file_path="out/data.csv")

    assert read_back == "a,b\n1,2\n"


def test_line_windows_work_on_a_remote_store(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store.files["/ws/big.log"] = "".join(f"L{i}\n" for i in range(1, 101))

    assert (
        FileReadTool()._run(file_path="big.log", start_line=3, line_count=2)
        == "L3\nL4\n"
    )


def test_store_containment_is_honoured(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = FileReadTool()._run(file_path="/etc/passwd")

    assert "Invalid file path" in result
    assert "base_dir" in result


def test_writer_containment_is_honoured(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = FileWriterTool()._run(
        filename="../escape.txt", content="x", overwrite=True
    )

    assert "Error" in result
    assert store.files == {}


def test_overwrite_false_is_reported_by_the_store(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool = FileWriterTool()

    assert "successfully written" in tool._run(filename="a.txt", content="one")
    assert "already exists" in tool._run(filename="a.txt", content="two")
    assert store.files["/ws/a.txt"] == "one"


def test_directory_that_is_a_file_reports_clearly(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store.files["/ws/notadir"] = "x"

    result = FileWriterTool()._run(
        filename="f.txt", directory="notadir", content="x", overwrite=True
    )

    assert "a file already exists where a directory is needed" in result


def test_factory_returning_none_falls_back_to_local():
    register_file_store_factory(lambda: None)
    try:
        assert isinstance(resolve_file_store(), LocalFileStore)
    finally:
        reset_file_store_factory()


def test_factory_that_raises_falls_back_to_local(caplog):
    """A broken integration must not take file I/O down with it."""

    def boom() -> FileStore | None:
        raise RuntimeError("backing service unreachable")

    register_file_store_factory(boom)
    try:
        assert isinstance(resolve_file_store(), LocalFileStore)
    finally:
        reset_file_store_factory()
    assert "falling back to the local filesystem" in caplog.text


def test_store_is_bound_once_per_tool(store, tmp_path, monkeypatch):
    """Swapping the factory mid-run must not move an existing tool's files."""
    monkeypatch.chdir(tmp_path)
    tool = FileWriterTool()
    other = MemoryFileStore(root="/other")
    register_file_store_factory(lambda: other)

    tool._run(filename="a.txt", content="one", overwrite=True)

    assert store.files == {"/ws/a.txt": "one"}
    assert other.files == {}


def test_writer_anchors_base_dir_through_the_store(store, tmp_path, monkeypatch):
    """base_dir must be normalised by the store, not by os.path.realpath.

    A remote store's paths are not filesystem paths, so anchoring with local
    semantics would compute a sandbox root that its own resolve()/
    resolve_within() do not agree with.
    """
    monkeypatch.chdir(tmp_path)

    tool = FileWriterTool(base_dir="scoped")

    # The memory store roots at /ws, so its normalisation is what must show up.
    assert tool.base_dir == "/ws/scoped"
    assert str(tmp_path) not in str(tool.base_dir)


def test_writer_base_dir_confines_writes_under_a_remote_store(
    store, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    tool = FileWriterTool(base_dir="scoped")

    assert "successfully written" in tool._run(
        filename="in.txt", content="x", overwrite=True
    )
    assert "/ws/scoped/in.txt" in store.files

    escaped = tool._run(filename="x.txt", directory="/ws/elsewhere", content="x")
    assert "Error" in escaped


def test_reader_and_writer_agree_on_a_relative_base_dir(store, tmp_path, monkeypatch):
    """Both tools have to land on the same sandbox root for the same input."""
    monkeypatch.chdir(tmp_path)

    assert FileWriterTool(base_dir="shared").base_dir == (
        FileReadTool(base_dir="shared").base_dir
    )


def test_os_error_message_does_not_leak_an_absolute_path(tmp_path, monkeypatch):
    """resolve_within wraps OSError, and the writer returns that text verbatim.

    str() on an OSError carries the absolute filename, so it has to be reduced
    to the reason before it reaches an agent.
    """
    monkeypatch.chdir(tmp_path)

    result = FileWriterTool()._run(
        filename="a\x00b.txt", content="x", overwrite=True
    )

    assert "Error" in result
    assert str(tmp_path) not in result
