import os
import shutil
import tarfile
import tempfile
import zipfile
from unittest.mock import patch

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


# --- Security: symlink content must not leak out of the allow-list ---


@pytest.fixture
def symlink_env():
    """A working dir (allow-listed) containing a normal file and a symlink that
    points to a secret file OUTSIDE the allow-list."""
    work_dir = tempfile.mkdtemp()
    secret_dir = tempfile.mkdtemp()  # deliberately NOT allow-listed
    secret_file = os.path.join(secret_dir, "secret.txt")
    with open(secret_file, "w") as f:
        f.write("TOP_SECRET_PRIVATE_KEY")

    src = os.path.join(work_dir, "src")
    os.makedirs(src)
    with open(os.path.join(src, "normal.txt"), "w") as f:
        f.write("safe content")
    os.symlink(secret_file, os.path.join(src, "leak.txt"))

    prev = os.environ.get("CREWAI_TOOLS_ALLOWED_DIRS")
    os.environ["CREWAI_TOOLS_ALLOWED_DIRS"] = work_dir
    yield {"work_dir": work_dir, "src": src, "secret_dir": secret_dir}
    if prev is None:
        os.environ.pop("CREWAI_TOOLS_ALLOWED_DIRS", None)
    else:
        os.environ["CREWAI_TOOLS_ALLOWED_DIRS"] = prev
    shutil.rmtree(work_dir, ignore_errors=True)
    shutil.rmtree(secret_dir, ignore_errors=True)


def test_zip_excludes_symlink_to_outside_file(tool, symlink_env):
    out = os.path.join(symlink_env["work_dir"], "archive.zip")
    result = tool._run(
        input_path=symlink_env["src"], output_path=out, format="zip", overwrite=True
    )
    assert "Successfully compressed" in result
    assert "skipped for safety" in result

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "normal.txt" in names
        assert "leak.txt" not in names
        blob = b"".join(zf.read(n) for n in names)
    assert b"TOP_SECRET_PRIVATE_KEY" not in blob


def test_tar_excludes_symlink_to_outside_file(tool, symlink_env):
    out = os.path.join(symlink_env["work_dir"], "archive.tar.gz")
    result = tool._run(
        input_path=symlink_env["src"],
        output_path=out,
        format="tar.gz",
        overwrite=True,
    )
    assert "Successfully compressed" in result
    assert "skipped for safety" in result

    with tarfile.open(out) as tf:
        members = tf.getnames()
        assert any(m.endswith("normal.txt") for m in members)
        assert not any(m.endswith("leak.txt") for m in members)
        assert all(not (tf.getmember(m).issym() or tf.getmember(m).islnk()) for m in members)
