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
    FileStoreError,
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


class FailingFileStore(MemoryFileStore):
    """A store where one named operation fails, the way a remote one can.

    The local filesystem cannot produce most of these — an unreachable
    endpoint has no local equivalent — so a store that only ever fails the
    way `open()` does would not exercise the tools' error paths at all.
    """

    def __init__(self, failing: str, exc: Exception | None = None) -> None:
        super().__init__()
        self.failing = failing
        self.exc = exc or FileStoreError("workspace endpoint unreachable")

    def _maybe_fail(self, name: str) -> None:
        if name == self.failing:
            raise self.exc

    def normalize(self, path: str, base_dir: str | None = None) -> str:
        self._maybe_fail("normalize")
        return super().normalize(path, base_dir)

    def resolve(self, path: str, base_dir: str | None = None) -> str:
        self._maybe_fail("resolve")
        return super().resolve(path, base_dir)

    def resolve_within(self, directory: str, filename: str) -> str:
        self._maybe_fail("resolve_within")
        return super().resolve_within(directory, filename)

    def display(self, resolved: str, base: str | None = None) -> str:
        self._maybe_fail("display")
        return super().display(resolved, base)

    def exists(self, resolved: str) -> bool:
        self._maybe_fail("exists")
        return super().exists(resolved)

    def ensure_parent(self, resolved: str) -> None:
        self._maybe_fail("ensure_parent")
        super().ensure_parent(resolved)

    def open_text(self, resolved: str, encoding: str):
        self._maybe_fail("open_text")
        return super().open_text(resolved, encoding)

    def write_text(
        self, resolved: str, content: str, encoding: str, *, overwrite: bool
    ) -> None:
        self._maybe_fail("write_text")
        super().write_text(resolved, content, encoding, overwrite=overwrite)


@pytest.fixture
def store():
    memory = MemoryFileStore()
    register_file_store_factory(lambda: memory)
    yield memory
    reset_file_store_factory()


@pytest.fixture
def failing_store():
    """Register a store whose *named* operation fails; the test picks which."""
    created: list[FailingFileStore] = []

    def _register(failing: str, exc: Exception | None = None) -> FailingFileStore:
        broken = FailingFileStore(failing, exc)
        register_file_store_factory(lambda: broken)
        created.append(broken)
        return broken

    yield _register
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


# --- reconstruction through pydantic -----------------------------------------
#
# BaseTool rebuilds a serialized tool with `model_validate` (see
# `_resolve_tool_dict`), which skips `__init__` entirely. Anything derived only
# in `__init__` comes back missing, and for the store that means every read
# raising AttributeError on a None.


def test_reconstructed_reader_reads_through_the_store(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store.files["/ws/notes.txt"] = "reconstructed\n"

    tool = FileReadTool.model_validate({"file_path": "notes.txt"})

    assert tool._store is store
    assert tool._run() == "reconstructed\n"


def test_reconstructed_writer_writes_through_the_store(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    tool = FileWriterTool.model_validate({"base_dir": "scoped"})

    assert tool._store is store
    assert tool.base_dir == "/ws/scoped"
    assert "successfully written" in tool._run(
        filename="a.txt", content="x", overwrite=True
    )
    assert store.files == {"/ws/scoped/a.txt": "x"}


def test_reconstructed_reader_keeps_its_declared_file(store, tmp_path, monkeypatch):
    """The declared path is pinned in the hook, so a rebuild pins it too."""
    monkeypatch.chdir(tmp_path)
    store.files["/ws/sub/pinned.txt"] = "kept\n"

    fresh = FileReadTool(file_path="sub/pinned.txt")
    rebuilt = FileReadTool.model_validate({"file_path": "sub/pinned.txt"})

    assert rebuilt._declared_realpath == fresh._declared_realpath == "/ws/sub/pinned.txt"
    assert rebuilt._declared_label == fresh._declared_label
    assert rebuilt.description == fresh.description
    # Addressable by the label the description shows, same as a fresh tool.
    assert rebuilt._run(file_path=rebuilt._declared_label) == "kept\n"


def test_reader_round_trips_through_model_dump(store, tmp_path, monkeypatch):
    """The full serialize/deserialize cycle, not just a hand-built dict."""
    monkeypatch.chdir(tmp_path)
    store.files["/ws/notes.txt"] = "dumped\n"

    original = FileReadTool(file_path="notes.txt")
    rebuilt = FileReadTool.model_validate(original.model_dump())

    assert rebuilt._run() == "dumped\n"


# --- a store that fails while a tool is being built --------------------------
#
# `normalize` and `display` are specified as pure and non-raising, precisely
# because the tools call them during construction — including when pydantic
# rebuilds a serialized crew. A store that breaks that contract anyway must not
# take a whole crew down with it, but the two derivations are not equal: the
# declared default file is a convenience, `base_dir` is a containment
# guarantee. So one degrades and one does not.


def test_a_store_failing_on_the_declared_file_still_builds_the_tool(
    failing_store, tmp_path, monkeypatch, caplog
):
    """Convenience: no default file rather than no crew."""
    monkeypatch.chdir(tmp_path)
    store = failing_store("display")

    tool = FileReadTool(file_path="notes.txt")

    assert tool._declared_realpath is None
    assert tool._declared_label is None
    assert "no default file" in caplog.text
    # Still a working tool — explicit paths go through the store as usual.
    store.failing = ""
    store.files["/ws/other.txt"] = "still working\n"
    assert tool._run(file_path="other.txt") == "still working\n"
    # And omitting the path reports the missing default rather than raising.
    assert "No file path provided" in tool._run()


def test_a_rebuild_that_loses_the_pin_stops_advertising_a_default(
    store, failing_store, tmp_path, monkeypatch
):
    """What the tool says must match what it does.

    The description is serialized, so on a rebuild it arrives already naming a
    default file. If the pin is then lost, leaving that text would tell the LLM
    it can omit `file_path` — and it would get "No file path provided" back.
    """
    monkeypatch.chdir(tmp_path)
    dumped = FileReadTool(file_path="notes.txt").model_dump()
    assert "The default file is" in dumped["description"]

    # Same tool, rebuilt against a store that can no longer label the path.
    failing_store("display")
    rebuilt = FileReadTool.model_validate(dumped)

    assert rebuilt._declared_label is None
    assert "The default file is" not in rebuilt.description
    assert rebuilt.description == FileReadTool.model_fields["description"].default
    assert "No file path provided" in rebuilt._run()


def test_a_store_failing_on_base_dir_is_not_swallowed(
    failing_store, tmp_path, monkeypatch
):
    """Containment: a sandbox root that cannot be anchored must not be faked.

    Degrading here would leave `base_dir` relative, so a later chdir could move
    the sandbox — a weaker guarantee than the caller asked for, arrived at
    silently. Failing loudly is the point.
    """
    monkeypatch.chdir(tmp_path)
    failing_store("normalize")

    with pytest.raises((FileStoreError, OSError)):
        FileReadTool(base_dir="scoped")

    with pytest.raises((FileStoreError, OSError)):
        FileWriterTool(base_dir="scoped")


# --- what the declared-path pin means across a rebuild -----------------------
#
# The pin is derived from what was *declared*, so whether it survives a rebuild
# in a different working directory depends on whether the declaration named
# somewhere absolute. These three cases are the whole story; they are pinned
# here so the behavior is a decision rather than an accident. Note the local
# store is the one that makes this observable, since its `normalize` is the one
# that consults the process cwd.


def test_an_absolute_declared_path_survives_a_rebuild_elsewhere(tmp_path, monkeypatch):
    """The strongest case: an absolute declaration is cwd-independent."""
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir(), there.mkdir()
    (here / "notes.txt").write_text("from here\n")
    (there / "notes.txt").write_text("from there\n")

    monkeypatch.chdir(here)
    dumped = FileReadTool(file_path=str(here / "notes.txt")).model_dump()
    monkeypatch.chdir(there)
    rebuilt = FileReadTool.model_validate(dumped)

    assert rebuilt._run() == "from here\n"


def test_a_declared_base_dir_pins_a_relative_path_across_a_rebuild(tmp_path, monkeypatch):
    """base_dir is anchored at construction, so it carries the pin with it."""
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir(), there.mkdir()
    (here / "notes.txt").write_text("from here\n")
    (there / "notes.txt").write_text("from there\n")

    monkeypatch.chdir(here)
    dumped = FileReadTool(file_path="notes.txt", base_dir=str(here)).model_dump()
    monkeypatch.chdir(there)
    rebuilt = FileReadTool.model_validate(dumped)

    assert rebuilt._run() == "from here\n"


def test_a_bare_relative_declared_path_reanchors_on_rebuild(tmp_path, monkeypatch):
    """A relative path with no base_dir names nothing absolute to preserve.

    It re-anchors to the rebuilding process's working directory — the same file
    the same arguments would name there. Documented rather than "fixed": the
    alternative is pinning a path from a working directory that, for a rebuild
    in a fresh container, no longer exists. Callers needing the pin to survive
    pass `base_dir`, which the test above covers.
    """
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir(), there.mkdir()
    (here / "notes.txt").write_text("from here\n")
    (there / "notes.txt").write_text("from there\n")

    monkeypatch.chdir(here)
    dumped = FileReadTool(file_path="notes.txt").model_dump()
    monkeypatch.chdir(there)
    rebuilt = FileReadTool.model_validate(dumped)

    assert rebuilt._run() == "from there\n"
    # And it is a real re-anchor, not a stale absolute path that happens to read.
    # resolve() because the store canonicalizes, and /tmp is a symlink on macOS.
    assert rebuilt._declared_realpath == str((there / "notes.txt").resolve())


# --- a store that fails ------------------------------------------------------
#
# A store may fail where the local filesystem never could. Every exit from
# these tools is an agent-visible string, so a store failure has to become one
# too rather than raising into the agent's step.


@pytest.mark.parametrize(
    "failing", ["resolve", "resolve_within", "display", "exists", "ensure_parent"]
)
def test_writer_reports_a_store_failure_instead_of_raising(
    failing, failing_store, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    failing_store(failing)

    result = FileWriterTool()._run(filename="a.txt", content="x", overwrite=True)

    assert "error occurred while writing" in result
    assert "memory" in result
    assert "workspace endpoint unreachable" in result


def test_writer_reports_a_failing_write(failing_store, tmp_path, monkeypatch):
    """write_text already had a handler; it must keep its own wording."""
    monkeypatch.chdir(tmp_path)
    failing_store("write_text")

    result = FileWriterTool()._run(filename="a.txt", content="x", overwrite=True)

    assert "An error occurred while writing to the file" in result
    assert "workspace endpoint unreachable" in result


@pytest.mark.parametrize("failing", ["resolve", "display"])
def test_reader_reports_a_store_failure_instead_of_raising(
    failing, failing_store, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    failing_store(failing)

    result = FileReadTool()._run(file_path="notes.txt")

    assert "Error" in result
    assert "memory" in result
    assert "workspace endpoint unreachable" in result


def test_reader_reports_a_failing_open(failing_store, tmp_path, monkeypatch):
    """open_text already had a handler; it must keep its own wording."""
    monkeypatch.chdir(tmp_path)
    failing_store("open_text")

    result = FileReadTool()._run(file_path="notes.txt")

    assert "Failed to read file" in result
    assert "workspace endpoint unreachable" in result


def _run_either(tool_cls, tool):
    """Drive whichever tool this is with its minimal arguments."""
    if tool_cls is FileReadTool:
        return tool._run(file_path="notes.txt")
    return tool._run(filename="a.txt", content="x", overwrite=True)


@pytest.mark.parametrize("tool_cls", [FileReadTool, FileWriterTool])
def test_an_os_error_from_the_store_is_reported_too(
    tool_cls, failing_store, tmp_path, monkeypatch
):
    """Not every store failure arrives as FileStoreError.

    A store wrapping a socket or a subprocess can raise OSError from resolve,
    which is neither a ValueError nor something the read/write handlers see.
    ``strerror`` is what survives redaction, so that is what an agent gets.
    """
    monkeypatch.chdir(tmp_path)
    failing_store("resolve", OSError(104, "Connection reset by peer"))

    result = _run_either(tool_cls, tool_cls())

    assert "Connection reset by peer" in result


@pytest.mark.parametrize("tool_cls", [FileReadTool, FileWriterTool])
def test_a_bare_os_error_degrades_to_its_type_without_raising(
    tool_cls, failing_store, tmp_path, monkeypatch
):
    """A single-arg OSError has no strerror, so only its type survives.

    That is `format_error_for_display` holding the line it was given in #6692:
    an OS-populated OSError renders its absolute filename into `str()`, so the
    message is never passed through wholesale. The cost is a thin report for a
    hand-raised `OSError("...")`; stores wanting a legible message should raise
    `FileStoreError`, whose text is preserved. What matters here is that the
    tool still returns rather than raising.
    """
    monkeypatch.chdir(tmp_path)
    failing_store("resolve", OSError("connection reset by peer"))

    result = _run_either(tool_cls, tool_cls())

    assert "OSError" in result
    assert "store failed" in result


def test_a_store_failure_does_not_leak_an_absolute_path(
    failing_store, tmp_path, monkeypatch
):
    """The store-failure path is agent-visible, so it gets the same redaction."""
    monkeypatch.chdir(tmp_path)
    failing_store("resolve", OSError(2, "No such file", str(tmp_path / "secret.txt")))

    result = FileWriterTool()._run(filename="a.txt", content="x", overwrite=True)

    assert str(tmp_path) not in result
