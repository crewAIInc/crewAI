import os
import shutil
import tempfile

import pytest

from crewai_tools.tools.file_writer_tool.file_writer_tool import FileWriterTool


@pytest.fixture
def tool():
    return FileWriterTool()


@pytest.fixture
def temp_env():
    temp_dir = tempfile.mkdtemp()
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
    result = tool._run(
        directory=temp_env["temp_dir"],
        content=temp_env["test_content"],
        overwrite=True,
    )
    assert "An error occurred while accessing key: 'filename'" in result


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
    assert read_file(path) == "Pre-existing content"
