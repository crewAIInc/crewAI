import os
from unittest.mock import patch
from zipfile import ZipFile

from crewai_tools.tools.files_compressor_tool.files_compressor_tool import (
    FileCompressorTool,
)
import pytest


@pytest.fixture
def tool():
    return FileCompressorTool()


@patch("os.path.exists", return_value=False)
def test_input_path_does_not_exist(mock_exists, tool):
    result = tool._run("nonexistent_path")
    assert "does not exist" in result


@patch("os.path.exists", return_value=True)
@patch("os.getcwd", return_value="/mocked/cwd")
@patch.object(FileCompressorTool, "_compress_zip")
@patch.object(FileCompressorTool, "_prepare_output", return_value=True)
def test_generate_output_path_default(
    mock_prepare, mock_compress, mock_cwd, mock_exists, tool
):
    result = tool._run(input_path="mydir", format="zip")
    assert "Successfully compressed" in result
    mock_compress.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch.object(FileCompressorTool, "_compress_zip")
@patch.object(FileCompressorTool, "_prepare_output", return_value=True)
def test_zip_compression(mock_prepare, mock_compress, mock_exists, tool):
    result = tool._run(
        input_path="some/path", output_path="archive.zip", format="zip", overwrite=True
    )
    assert "Successfully compressed" in result
    mock_compress.assert_called_once()


@patch("os.path.exists", return_value=True)
@patch.object(FileCompressorTool, "_compress_tar")
@patch.object(FileCompressorTool, "_prepare_output", return_value=True)
def test_tar_gz_compression(mock_prepare, mock_compress, mock_exists, tool):
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
    result = tool._run(
        input_path="some/path", output_path=f"archive.{format}", format=format
    )
    assert "not supported" in result


@patch("os.path.exists", return_value=True)
def test_extension_mismatch(_, tool):
    result = tool._run(
        input_path="some/path", output_path="archive.zip", format="tar.gz"
    )
    assert "must have a '.tar.gz' extension" in result


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("os.path.exists", return_value=True)
def test_existing_output_no_overwrite(_, __, ___, tool):
    result = tool._run(
        input_path="some/path", output_path="archive.zip", format="zip", overwrite=False
    )
    assert "overwrite is set to False" in result


@patch("os.path.exists", return_value=True)
@patch("zipfile.ZipFile", side_effect=PermissionError)
def test_permission_error(mock_zip, _, tool):
    result = tool._run(
        input_path="file.txt", output_path="file.zip", format="zip", overwrite=True
    )
    assert "Permission denied" in result


@patch("os.path.exists", return_value=True)
@patch("zipfile.ZipFile", side_effect=FileNotFoundError)
def test_file_not_found_during_zip(mock_zip, _, tool):
    result = tool._run(
        input_path="file.txt", output_path="file.zip", format="zip", overwrite=True
    )
    assert "File not found" in result


@patch("os.path.exists", return_value=True)
@patch("zipfile.ZipFile", side_effect=Exception("Unexpected"))
def test_general_exception_during_zip(mock_zip, _, tool):
    result = tool._run(
        input_path="file.txt", output_path="file.zip", format="zip", overwrite=True
    )
    assert "unexpected error" in result


# Test: Output directory is created when missing
@patch("os.makedirs")
@patch("os.path.exists", return_value=False)
def test_prepare_output_makes_dir(mock_exists, mock_makedirs):
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
