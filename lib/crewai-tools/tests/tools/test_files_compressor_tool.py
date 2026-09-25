import os
import tarfile
from contextlib import contextmanager
from unittest.mock import patch
from zipfile import ZipFile

from crewai_tools.tools.files_compressor_tool.files_compressor_tool import (
    FileCompressorTool,
)
import pytest


@pytest.fixture
def tool():
    """Return a fresh file compressor tool for each test."""
    return FileCompressorTool()


@patch("os.path.exists", return_value=False)
def test_input_path_does_not_exist(mock_exists, tool):
    """Report a missing input path without attempting compression."""
    result = tool._run("nonexistent_path")
    assert "does not exist" in result


@patch("os.path.exists", return_value=True)
@patch("os.getcwd", return_value="/mocked/cwd")
@patch.object(FileCompressorTool, "_compress_zip")
@patch.object(FileCompressorTool, "_prepare_output", return_value=True)
def test_generate_output_path_default(
    mock_prepare, mock_compress, mock_cwd, mock_exists, tool
):
    """Generate an output path when the caller does not provide one."""
    result = tool._run(input_path="mydir", format="zip")
    assert "Successfully compressed" in result
    mock_compress.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch.object(FileCompressorTool, "_compress_zip")
@patch.object(FileCompressorTool, "_prepare_output", return_value=True)
def test_zip_compression(mock_prepare, mock_compress, mock_exists, tool):
    """Dispatch ZIP requests to the ZIP compressor."""
    result = tool._run(
        input_path="some/path", output_path="archive.zip", format="zip", overwrite=True
    )
    assert "Successfully compressed" in result
    mock_compress.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch.object(FileCompressorTool, "_compress_tar")
@patch.object(FileCompressorTool, "_prepare_output", return_value=True)
def test_tar_gz_compression(mock_prepare, mock_compress, mock_exists, tool):
    """Dispatch gzip-compressed TAR requests to the TAR compressor."""
    result = tool._run(
        input_path="some/path",
        output_path="archive.tar.gz",
        format="tar.gz",
        overwrite=True,
    )
    assert "Successfully compressed" in result
    mock_compress.assert_called_once()


@pytest.mark.parametrize("format", ["tar", "tar.bz2", "tar.xz"])
@patch("os.path.exists", return_value=True)
@patch.object(FileCompressorTool, "_compress_tar")
@patch.object(FileCompressorTool, "_prepare_output", return_value=True)
def test_other_tar_formats(mock_prepare, mock_compress, mock_exists, format, tool):
    """Dispatch each supported TAR variant to the TAR compressor."""
    result = tool._run(
        input_path="path/to/input",
        output_path=f"archive.{format}",
        format=format,
        overwrite=True,
    )
    assert "Successfully compressed" in result
    mock_compress.assert_called_once()


@pytest.mark.parametrize("format", ["rar", "7z"])
@patch("os.path.exists", return_value=True)  # Ensure input_path exists
def test_unsupported_format(_, tool, format):
    """Reject archive formats that the tool does not support."""
    result = tool._run(
        input_path="some/path", output_path=f"archive.{format}", format=format
    )
    assert "not supported" in result


@patch("os.path.exists", return_value=True)
def test_extension_mismatch(_, tool):
    """Reject output extensions that do not match the selected format."""
    result = tool._run(
        input_path="some/path", output_path="archive.zip", format="tar.gz"
    )
    assert "must have a '.tar.gz' extension" in result


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("os.path.exists", return_value=True)
def test_existing_output_no_overwrite(_, __, ___, tool):
    """Keep an existing archive when overwrite is disabled."""
    result = tool._run(
        input_path="some/path", output_path="archive.zip", format="zip", overwrite=False
    )
    assert "overwrite is set to False" in result


@patch("os.path.exists", return_value=True)
@patch("zipfile.ZipFile", side_effect=PermissionError)
def test_permission_error(mock_zip, _, tool, tmp_path, monkeypatch):
    """Return a clear message when the archive cannot be opened for writing."""
    monkeypatch.chdir(tmp_path)
    result = tool._run(
        input_path="file.txt", output_path="file.zip", format="zip", overwrite=True
    )
    assert "Permission denied" in result


@patch("os.path.exists", return_value=True)
@patch("zipfile.ZipFile", side_effect=FileNotFoundError)
def test_file_not_found_during_zip(mock_zip, _, tool, tmp_path, monkeypatch):
    """Return a clear message when a source disappears during compression."""
    monkeypatch.chdir(tmp_path)
    result = tool._run(
        input_path="file.txt", output_path="file.zip", format="zip", overwrite=True
    )
    assert "File not found" in result


@patch("os.path.exists", return_value=True)
@patch("zipfile.ZipFile", side_effect=Exception("Unexpected"))
def test_general_exception_during_zip(mock_zip, _, tool, tmp_path, monkeypatch):
    """Report unexpected ZIP errors without propagating them."""
    monkeypatch.chdir(tmp_path)
    result = tool._run(
        input_path="file.txt", output_path="file.zip", format="zip", overwrite=True
    )
    assert "unexpected error" in result


# Test: Output directory is created when missing
@patch("os.makedirs")
@patch("os.path.exists", return_value=False)
def test_prepare_output_makes_dir(mock_exists, mock_makedirs):
    """Create a missing parent directory before writing an archive."""
    tool = FileCompressorTool()
    result = tool._prepare_output("some/missing/path/file.zip", overwrite=True)
    assert result is True
    mock_makedirs.assert_called_once()


# The tests below drive the real ``zipfile`` path on purpose: everything above mocks
# ``zipfile.ZipFile``, which is why a self-referential archive went unnoticed.
def test_zip_output_inside_input_is_not_archived(tmp_path, monkeypatch, tool):
    """The archive must not contain itself when it is written into the source tree.

    ``ZipFile(output_path, "w")`` creates the output before ``os.walk`` enumerates the directory,
    so without an explicit skip the archive becomes an empty member of itself — a file that was
    never in the source directory, and one that extracts to an empty archive.
    """
    monkeypatch.chdir(tmp_path)  # validate_file_path allows the cwd tree
    (tmp_path / "payload.txt").write_text("hello", encoding="utf-8")

    result = tool._run(input_path=".", output_path="bundle.zip")

    assert "Successfully compressed" in result
    with ZipFile(tmp_path / "bundle.zip") as archive:
        assert archive.namelist() == ["payload.txt"]


def test_zip_output_in_nested_input_directory_is_not_archived(
    tmp_path, monkeypatch, tool
):
    """The skip matches the file, not its archive name, so a nested output is skipped too."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "payload.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "nested").mkdir()

    result = tool._run(input_path=".", output_path="nested/bundle.zip")

    assert "Successfully compressed" in result
    with ZipFile(tmp_path / "nested" / "bundle.zip") as archive:
        names = archive.namelist()

    assert "nested/bundle.zip" not in names
    assert "payload.txt" in names


def test_zip_keeps_sibling_archives_in_input(tmp_path, monkeypatch, tool):
    """Only the archive being written is skipped — an unrelated .zip in the tree is still kept."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "payload.txt").write_text("hello", encoding="utf-8")

    other = tmp_path / "existing.zip"
    with ZipFile(other, "w") as seed:
        seed.writestr("inner.txt", "inner")

    result = tool._run(input_path=".", output_path="bundle.zip")

    assert "Successfully compressed" in result
    with ZipFile(tmp_path / "bundle.zip") as archive:
        names = archive.namelist()

    assert sorted(names) == ["existing.zip", "payload.txt"]


def test_zip_overwrites_an_archive_that_already_sits_in_input(
    tmp_path, monkeypatch, tool
):
    """Replacing the archive in place is an ordinary overwrite, not a self-alias to reject."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "payload.txt").write_text("hello", encoding="utf-8")

    with ZipFile(tmp_path / "bundle.zip", "w") as stale:
        stale.writestr("stale.txt", "old")

    result = tool._run(input_path=".", output_path="bundle.zip", overwrite=True)

    assert "Successfully compressed" in result
    with ZipFile(tmp_path / "bundle.zip") as archive:
        assert archive.namelist() == ["payload.txt"]


def test_zip_output_hard_linked_to_an_input_file_is_rejected(
    tmp_path, monkeypatch, tool
):
    """A hard link shares the output's inode, so opening it would truncate the source file.

    Path comparison cannot see this — ``payload.txt`` and ``bundle.zip`` are different paths — so
    the output has to be rejected by filesystem identity *before* ``ZipFile`` opens it.
    """
    monkeypatch.chdir(tmp_path)
    payload = tmp_path / "payload.txt"
    payload.write_text("hello", encoding="utf-8")
    os.link(payload, tmp_path / "bundle.zip")

    result = tool._run(input_path=".", output_path="bundle.zip", overwrite=True)

    assert "Successful" not in result
    assert "same file" in result
    # The source file must still be intact: this is the damage the guard prevents.
    assert payload.read_text(encoding="utf-8") == "hello"


def test_zip_input_and_output_same_file_is_rejected(tmp_path, monkeypatch, tool):
    """Compressing a file onto itself truncates it before it can be read."""
    monkeypatch.chdir(tmp_path)
    payload = tmp_path / "payload.zip"
    payload.write_text("hello", encoding="utf-8")

    result = tool._run(input_path="payload.zip", output_path="payload.zip", overwrite=True)

    assert "Successful" not in result
    assert payload.read_text(encoding="utf-8") == "hello"


def test_zip_output_symlinked_from_input_is_rejected(tmp_path, monkeypatch, tool):
    """A symlink resolving to the output must not be exempted as "the output itself".

    ``realpath`` equates the symlink with its target, but they are not the same source: opening the
    output truncates the target, which is an input file. Only the output's own path is exempt.
    """
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data.zip"
    with ZipFile(data, "w") as seed:
        seed.writestr("inner.txt", "inner")
    try:
        (tmp_path / "alias.zip").symlink_to(data)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this host")

    result = tool._run(input_path=".", output_path="alias.zip", overwrite=True)

    assert "Successful" not in result
    assert "same file" in result
    # The symlink's target is an input file, so it has to survive the attempt.
    with ZipFile(data) as archive:
        assert archive.namelist() == ["inner.txt"]


def test_tar_output_hard_linked_to_an_input_file_is_rejected(
    tmp_path, monkeypatch, tool
):
    """The overlap guard is format-independent: ``tarfile.open`` truncates at open too.

    ``tarfile`` protects the archive from being added to itself, which is a different problem from
    an output that *is* an input file — it cannot help here, because the truncation happens before
    the walk.
    """
    monkeypatch.chdir(tmp_path)
    payload = tmp_path / "payload.txt"
    payload.write_text("hello", encoding="utf-8")
    os.link(payload, tmp_path / "bundle.tar.gz")

    result = tool._run(
        input_path=".", output_path="bundle.tar.gz", format="tar.gz", overwrite=True
    )

    assert "Successful" not in result
    assert "same file" in result
    assert payload.read_text(encoding="utf-8") == "hello"


@pytest.mark.parametrize("archive_format", ["tar", "tar.gz", "tar.bz2", "tar.xz"])
def test_tar_compression_reuses_validated_output_descriptor(
    tmp_path, monkeypatch, tool, archive_format
):
    """Write every TAR format through the descriptor validated before truncation."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.txt").write_text("hello", encoding="utf-8")
    output = tmp_path / f"bundle.{archive_format}"

    validated_outputs = []
    real_open_validated_output = FileCompressorTool._open_validated_output

    @contextmanager
    def record_validated_output(*args, **kwargs):
        with real_open_validated_output(*args, **kwargs) as output_file:
            validated_outputs.append(output_file)
            yield output_file

    monkeypatch.setattr(
        FileCompressorTool,
        "_open_validated_output",
        record_validated_output,
    )
    with patch(
        "crewai_tools.tools.files_compressor_tool.files_compressor_tool.tarfile.open",
        wraps=tarfile.open,
    ) as tar_open:
        result = tool._run(
            input_path="source",
            output_path=output.name,
            format=archive_format,
        )

    assert "Successfully compressed" in result
    assert tar_open.call_args.kwargs["fileobj"] is validated_outputs[0]
    with tarfile.open(output, "r:*") as archive:
        assert "source/payload.txt" in archive.getnames()


def test_zip_output_symlink_outside_input_pointing_into_it_is_rejected(
    tmp_path, monkeypatch, tool
):
    """An output symlink outside the tree can still point at an in-tree input file.

    ``validate_file_path`` resolves the caller's output path before the overlap guard runs, so the
    guard only ever sees the in-tree target and would exempt it as "the output". Which path the
    caller named is what decides that exemption, not what it resolves to.
    """
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    payload = source / "payload.zip"
    with ZipFile(payload, "w") as seed:
        seed.writestr("inner.txt", "inner")
    try:
        (outside / "bundle.zip").symlink_to(payload)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this host")

    result = tool._run(
        input_path="source", output_path="outside/bundle.zip", overwrite=True
    )

    assert "Successful" not in result
    assert "same file" in result
    with ZipFile(payload) as archive:
        assert archive.namelist() == ["inner.txt"]


def test_existing_output_exemption_honors_filesystem_case_rules(
    tmp_path, monkeypatch, tool
):
    """The lexical exemption follows the host case-normalization rules."""
    output = tmp_path / "bundle.zip"
    output.write_bytes(b"stale")

    # Model a case-insensitive filesystem while keeping this test portable.
    monkeypatch.setattr(os.path, "normcase", str.lower)

    output_fd = os.open(output, os.O_RDWR)
    try:
        tool._reject_output_aliasing_input(
            str(tmp_path),
            str(output),
            str(tmp_path / "BUNDLE.zip"),
            output_fd,
        )
    finally:
        os.close(output_fd)


@pytest.mark.parametrize(
    ("archive_format", "output_name"),
    [("zip", "bundle.zip"), ("tar.gz", "bundle.tar.gz")],
)
def test_output_path_swap_before_open_cannot_truncate_input(
    tmp_path, monkeypatch, tool, archive_format, output_name
):
    """Validate the descriptor opened after a hostile path replacement before truncating it."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "payload.txt"
    payload.write_text("keep me", encoding="utf-8")
    output = tmp_path / output_name
    output.write_bytes(b"stale")

    real_open = os.open

    swapped = False

    def swap_output_then_open(path, flags, mode=0o777, *, dir_fd=None):
        """Replace the checked output path immediately before the production open."""
        nonlocal swapped
        opens_output = (dir_fd is not None and path == output.name) or (
            dir_fd is None and os.path.abspath(path) == str(output)
        )
        if opens_output and not swapped:
            swapped = True
            output.unlink()
            try:
                output.symlink_to(payload)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks are not available on this host")

        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_output_then_open)

    result = tool._run(
        input_path="source",
        output_path=output_name,
        overwrite=True,
        format=archive_format,
    )

    assert "Successful" not in result
    assert "symbolic-link path" in result
    assert payload.read_text(encoding="utf-8") == "keep me"


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd,
    reason="descriptor-relative no-follow opens are unavailable",
)
def test_output_parent_swap_cannot_redirect_archive_write(
    tmp_path, monkeypatch, tool
):
    """Pin each parent directory so replacing its path cannot redirect the archive write."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.txt").write_text("hello", encoding="utf-8")

    intended_parent = tmp_path / "intended"
    intended_parent.mkdir()
    (intended_parent / "bundle.zip").write_bytes(b"old archive")
    replacement_parent = tmp_path / "replacement"
    replacement_parent.mkdir()
    replacement_output = replacement_parent / "bundle.zip"
    replacement_output.write_bytes(b"must stay intact")
    pinned_parent = tmp_path / "pinned"

    real_open = os.open
    swapped = False

    def swap_parent_after_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        opened_fd = (
            real_open(path, flags, mode)
            if dir_fd is None
            else real_open(path, flags, mode, dir_fd=dir_fd)
        )
        if path == intended_parent.name and dir_fd is not None and not swapped:
            swapped = True
            intended_parent.rename(pinned_parent)
            replacement_parent.rename(intended_parent)
        return opened_fd

    monkeypatch.setattr(os, "open", swap_parent_after_open)

    result = tool._run(
        input_path="source",
        output_path="intended/bundle.zip",
        overwrite=True,
    )

    assert "Successfully compressed" in result
    assert (intended_parent / "bundle.zip").read_bytes() == b"must stay intact"
    with ZipFile(pinned_parent / "bundle.zip") as archive:
        assert archive.namelist() == ["payload.txt"]
