import os
import tempfile

import pytest

from crewai_tools.tools.file_read_tool.file_read_tool import FileReadTool


@pytest.fixture
def tool():
    return FileReadTool()


@pytest.fixture
def temp_file():
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(content)

    yield {"path": path, "content": content}

    os.unlink(path)


def test_read_entire_file(tool, temp_file):
    result = tool._run(file_path=temp_file["path"])
    assert result == temp_file["content"]


def test_read_file_with_constructor_path(temp_file):
    tool = FileReadTool(file_path=temp_file["path"])
    result = tool._run()
    assert result == temp_file["content"]


def test_read_with_start_line(tool, temp_file):
    result = tool._run(file_path=temp_file["path"], start_line=3)
    assert result == "Line 3\nLine 4\nLine 5\n"


def test_read_with_line_count(tool, temp_file):
    result = tool._run(file_path=temp_file["path"], start_line=2, line_count=2)
    assert result == "Line 2\nLine 3\n"


def test_file_not_found(tool):
    result = tool._run(file_path="/nonexistent/path/file.txt")
    assert "Error: File not found" in result


def test_no_file_path_provided(tool):
    result = tool._run()
    assert "Error: No file path provided" in result


def test_start_line_exceeds_file_length(tool, temp_file):
    result = tool._run(file_path=temp_file["path"], start_line=100)
    assert "Error: Start line" in result


def test_read_single_line(tool, temp_file):
    result = tool._run(file_path=temp_file["path"], start_line=1, line_count=1)
    assert result == "Line 1\n"
