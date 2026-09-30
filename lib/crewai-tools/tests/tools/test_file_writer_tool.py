import os
import shutil
import tempfile

from crewai_tools.tools.file_writer_tool.file_writer_tool import FileWriterTool
import pytest


@pytest.fixture
def tool():
    return FileWriterTool()


@pytest.fixture
def temp_env(monkeypatch):
    # Writes are confined to the working directory, so make the temp dir the cwd.
    temp_dir = os.path.realpath(tempfile.mkdtemp())
    monkeypatch.chdir(temp_dir)
    test_file = "test.txt"
    test_content = "Hello, World!"

    yield {
        "temp_dir": temp_dir,
        "test_file": test_file,
        "test_content": test_content,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def get_test_path(filename, directory):
    return os.path.join(directory, filename)


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def test_basic_file_write(tool, temp_env):
    result = tool._run(
        filename=temp_env["test_file"],
        directory=temp_env["temp_dir"],
        content=temp_env["test_content"],
        overwrite=True,
    )

    path = get_test_path(temp_env["test_file"], temp_env["temp_dir"])
    assert os.path.exists(path)
    assert read_file(path) == temp_env["test_content"]
    assert "successfully written" in result
    assert temp_env["test_file"] in result
    assert temp_env["temp_dir"] not in result


def test_directory_creation(tool, temp_env):
    new_dir = os.path.join(temp_env["temp_dir"], "nested_dir")
    result = tool._run(
        filename=temp_env["test_file"],
        directory=new_dir,
        content=temp_env["test_content"],
        overwrite=True,
    )

    path = get_test_path(temp_env["test_file"], new_dir)
    assert os.path.exists(new_dir)
    assert os.path.exists(path)
    assert "successfully written" in result
    assert temp_env["test_file"] in result
    assert new_dir not in result


@pytest.mark.parametrize(
    "overwrite",
    ["y", "yes", "t", "true", "on", "1", True],
)
def test_overwrite_true(tool, temp_env, overwrite):
    path = get_test_path(temp_env["test_file"], temp_env["temp_dir"])
    with open(path, "w") as f:
        f.write("Original content")

    result = tool._run(
        filename=temp_env["test_file"],
        directory=temp_env["temp_dir"],
        content="New content",
        overwrite=overwrite,
    )

    assert read_file(path) == "New content"
    assert "successfully written" in result


def test_invalid_overwrite_value(tool, temp_env):
    result = tool._run(
        filename=temp_env["test_file"],
        directory=temp_env["temp_dir"],
        content=temp_env["test_content"],
        overwrite="invalid",
    )
    assert "invalid value" in result


def test_missing_required_fields(tool, temp_env):
    with pytest.raises(TypeError, match="filename"):
        tool._run(
            directory=temp_env["temp_dir"],
            content=temp_env["test_content"],
            overwrite=True,
        )


def test_missing_required_fields_via_run(tool, temp_env):
    """The public entry point fails schema validation before reaching _run."""
    with pytest.raises(ValueError, match="validation failed"):
        tool.run(
            directory=temp_env["temp_dir"],
            content=temp_env["test_content"],
            overwrite=True,
        )


def test_documented_positional_signature(tool, temp_env):
    """The documented (filename, content, directory) call order must work."""
    result = tool._run("example.txt", "This is a test content.", "test_directory")

    assert "successfully written" in result
    assert read_file(os.path.join("test_directory", "example.txt")) == (
        "This is a test content."
    )


def test_defaults_applied_through_run(tool, temp_env):
    """overwrite/directory defaults come from the schema, not from _run kwargs."""
    assert "successfully written" in tool.run(filename="a.txt", content="one")
    assert "already exists" in tool.run(filename="a.txt", content="two")
    assert read_file("a.txt") == "one"


def test_empty_content(tool, temp_env):
    result = tool._run(
        filename=temp_env["test_file"],
        directory=temp_env["temp_dir"],
        content="",
        overwrite=True,
    )

    path = get_test_path(temp_env["test_file"], temp_env["temp_dir"])
    assert os.path.exists(path)
    assert read_file(path) == ""
    assert "successfully written" in result


@pytest.mark.parametrize(
    "overwrite",
    ["n", "no", "f", "false", "off", "0", False],
)
def test_file_exists_error_handling(tool, temp_env, overwrite):
    path = get_test_path(temp_env["test_file"], temp_env["temp_dir"])
    with open(path, "w") as f:
        f.write("Pre-existing content")

    result = tool._run(
        filename=temp_env["test_file"],
        directory=temp_env["temp_dir"],
        content="Should not be written",
        overwrite=overwrite,
    )

    assert "already exists and overwrite option was not passed" in result
    assert temp_env["test_file"] in result
    assert temp_env["temp_dir"] not in result
    assert read_file(path) == "Pre-existing content"



def test_blocks_traversal_in_filename(tool, temp_env):
    # Create a sibling "outside" directory so we can assert nothing was written there.
    outside_dir = tempfile.mkdtemp()
    outside_file = os.path.join(outside_dir, "outside.txt")
    try:
        result = tool._run(
            filename=f"../{os.path.basename(outside_dir)}/outside.txt",
            directory=temp_env["temp_dir"],
            content="should not be written",
            overwrite=True,
        )
        assert "Error" in result
        assert not os.path.exists(outside_file)
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_blocks_absolute_path_in_filename(tool, temp_env):
    # Use a temp file outside temp_dir as the absolute target so we don't
    # depend on /etc/passwd existing or being writable on the host.
    outside_dir = tempfile.mkdtemp()
    outside_file = os.path.join(outside_dir, "target.txt")
    try:
        result = tool._run(
            filename=outside_file,
            directory=temp_env["temp_dir"],
            content="should not be written",
            overwrite=True,
        )
        assert "Error" in result
        assert not os.path.exists(outside_file)
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_blocks_symlink_escape(tool, temp_env):
    # Symlink inside temp_dir pointing to a separate temp "outside" directory.
    outside_dir = tempfile.mkdtemp()
    outside_file = os.path.join(outside_dir, "target.txt")
    link = os.path.join(temp_env["temp_dir"], "escape")
    os.symlink(outside_dir, link)
    try:
        result = tool._run(
            filename="escape/target.txt",
            directory=temp_env["temp_dir"],
            content="should not be written",
            overwrite=True,
        )
        assert "Error" in result
        assert not os.path.exists(outside_file)
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_blocks_directory_outside_working_directory(tool, temp_env):
    """An absolute 'directory' must not escape the working directory."""
    outside_dir = tempfile.mkdtemp()
    outside_file = os.path.join(outside_dir, "pwned.txt")
    try:
        result = tool._run(
            filename="pwned.txt",
            directory=outside_dir,
            content="should not be written",
            overwrite=True,
        )
        assert "Error" in result
        assert not os.path.exists(outside_file)
        assert outside_dir not in result
        # Point users at base_dir, not the process-wide escape hatch, which
        # would also disable the SSRF checks on URL-fetching tools.
        assert "base_dir" in result
        assert "CREWAI_TOOLS_ALLOW_UNSAFE_PATHS" not in result
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_blocks_directory_traversal_outside_working_directory(tool, temp_env):
    """'..' in the directory must not escape the working directory either."""
    outside_dir = tempfile.mkdtemp()
    outside_file = os.path.join(outside_dir, "pwned.txt")
    relative = os.path.join("..", os.path.basename(outside_dir))
    try:
        result = tool._run(
            filename="pwned.txt",
            directory=relative,
            content="should not be written",
            overwrite=True,
        )
        assert "Error" in result
        assert not os.path.exists(outside_file)
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_base_dir_widens_the_sandbox(temp_env):
    """base_dir lets a developer authorize writes outside the working directory."""
    outside_dir = tempfile.mkdtemp()
    try:
        scoped = FileWriterTool(base_dir=outside_dir)
        result = scoped._run(
            filename="allowed.txt",
            directory=outside_dir,
            content="written",
            overwrite=True,
        )
        assert "successfully written" in result
        assert read_file(os.path.join(outside_dir, "allowed.txt")) == "written"
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_base_dir_anchors_relative_directories(temp_env):
    """With base_dir set, a relative (or omitted) directory resolves under it."""
    outside_dir = tempfile.mkdtemp()
    try:
        scoped = FileWriterTool(base_dir=outside_dir)

        assert "successfully written" in scoped._run(
            filename="top.txt", content="top", overwrite=True
        )
        assert "successfully written" in scoped._run(
            filename="deep.txt", directory="nested", content="deep", overwrite=True
        )

        assert read_file(os.path.join(outside_dir, "top.txt")) == "top"
        assert read_file(os.path.join(outside_dir, "nested", "deep.txt")) == "deep"
        # The working directory must not have been touched.
        assert not os.path.exists(os.path.join(temp_env["temp_dir"], "top.txt"))
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_relative_base_dir_is_anchored_at_construction(temp_env, monkeypatch):
    """A relative base_dir must not follow a later chdir."""
    allowed = os.path.join(temp_env["temp_dir"], "allowed")
    os.makedirs(allowed)
    nested = os.path.join(temp_env["temp_dir"], "sub")
    os.makedirs(nested)

    scoped = FileWriterTool(base_dir="allowed")
    assert scoped.base_dir == os.path.realpath(allowed)

    monkeypatch.chdir(nested)
    result = scoped._run(filename="x.txt", content="written", overwrite=True)

    assert "successfully written" in result
    assert read_file(os.path.join(allowed, "x.txt")) == "written"


def test_base_dir_survives_a_serialization_round_trip(temp_env):
    outside = tempfile.mkdtemp()
    try:
        restored = FileWriterTool.model_validate(
            FileWriterTool(base_dir=outside).model_dump()
        )
        assert restored.base_dir == os.path.realpath(outside)
        assert "successfully written" in restored._run(
            filename="x.txt", directory=outside, content="written", overwrite=True
        )
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_base_dir_still_blocks_escapes(temp_env):
    """base_dir moves the sandbox; it does not remove it."""
    allowed_dir = tempfile.mkdtemp()
    outside_dir = tempfile.mkdtemp()
    outside_file = os.path.join(outside_dir, "pwned.txt")
    try:
        scoped = FileWriterTool(base_dir=allowed_dir)
        result = scoped._run(
            filename="pwned.txt",
            directory=outside_dir,
            content="should not be written",
            overwrite=True,
        )
        assert "Error" in result
        assert not os.path.exists(outside_file)
    finally:
        shutil.rmtree(allowed_dir, ignore_errors=True)
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_escape_hatch_allows_writes_outside_working_directory(
    tool, temp_env, monkeypatch
):
    monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")
    outside_dir = tempfile.mkdtemp()
    try:
        result = tool._run(
            filename="allowed.txt",
            directory=outside_dir,
            content="written",
            overwrite=True,
        )
        assert "successfully written" in result
        assert read_file(os.path.join(outside_dir, "allowed.txt")) == "written"
    finally:
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_creates_parents_for_nested_filename(tool, temp_env):
    """Subdirectories in 'filename' are created even without a 'directory'."""
    result = tool._run(
        filename=os.path.join("sub", "dir", "x.txt"),
        content=temp_env["test_content"],
        overwrite=True,
    )

    assert "successfully written" in result
    assert read_file(os.path.join("sub", "dir", "x.txt")) == temp_env["test_content"]


def test_directory_that_is_an_existing_file(tool, temp_env):
    """The error must describe the real problem, not a bogus overwrite failure."""
    with open("notadir", "w") as f:
        f.write("x")

    result = tool._run(
        filename="f.txt",
        directory="notadir",
        content=temp_env["test_content"],
        overwrite=True,
    )

    assert "Error" in result
    assert "a file already exists where a directory is needed" in result
    assert "overwrite" not in result


@pytest.mark.parametrize(
    ("filename", "directory"),
    [("a\x00b.txt", "./"), ("ok.txt", "d\x00ir")],
)
def test_null_byte_returns_error_instead_of_raising(tool, filename, directory):
    """_run's contract is to return a descriptive string for bad input."""
    result = tool._run(
        filename=filename, directory=directory, content="x", overwrite=True
    )

    assert "Error" in result


def test_writes_utf8_by_default(tool, temp_env):
    content = "café — 日本語 — 🚀"
    tool._run(filename=temp_env["test_file"], content=content, overwrite=True)

    with open(temp_env["test_file"], "rb") as f:
        assert f.read() == content.encode("utf-8")


def test_encoding_is_configurable(temp_env):
    content = "café"
    FileWriterTool(encoding="latin-1")._run(
        filename=temp_env["test_file"], content=content, overwrite=True
    )

    with open(temp_env["test_file"], "rb") as f:
        assert f.read() == content.encode("latin-1")
