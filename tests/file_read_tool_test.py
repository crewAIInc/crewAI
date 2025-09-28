import os
from unittest.mock import mock_open, patch

from crewai_tools import FileReadTool


def test_file_read_tool_constructor():
    """Test FileReadTool initialization with file_path."""
    # Create a temporary test file
    test_file = "/tmp/test_file.txt"
    test_content = "Hello, World!"
    with open(test_file, "w") as f:
        f.write(test_content)

    # Test initialization with file_path
    tool = FileReadTool(file_path=test_file)
    assert tool.file_path == test_file
    assert "test_file.txt" in tool.description

    # Clean up
    os.remove(test_file)


def test_file_read_tool_run():
    """Test FileReadTool _run method with file_path at runtime."""
    test_file = "/tmp/test_file.txt"
    test_content = "Hello, World!"

    # Use mock_open to mock file operations
    with patch("builtins.open", mock_open(read_data=test_content)):
        # Test reading file with runtime file_path
        tool = FileReadTool()
        result = tool._run(file_path=test_file)
        assert result == test_content


def test_file_read_tool_error_handling():
    """Test FileReadTool error handling."""
    # Test missing file path
    tool = FileReadTool()
    result = tool._run()
    assert "Error: No file path provided" in result

    # Test non-existent file
    result = tool._run(file_path="/nonexistent/file.txt")
    assert "Error: File not found at path:" in result

    # Test permission error
    with patch("builtins.open", side_effect=PermissionError()):
        result = tool._run(file_path="/tmp/no_permission.txt")
        assert "Error: Permission denied" in result


def test_file_read_tool_constructor_and_run():
    """Test FileReadTool using both constructor and runtime file paths."""
    test_file1 = "/tmp/test1.txt"
    test_file2 = "/tmp/test2.txt"
    content1 = "File 1 content"
    content2 = "File 2 content"

    # First test with content1
    with patch("builtins.open", mock_open(read_data=content1)):
        tool = FileReadTool(file_path=test_file1)
        result = tool._run()
        assert result == content1

    # Then test with content2 (should override constructor file_path)
    with patch("builtins.open", mock_open(read_data=content2)):
        result = tool._run(file_path=test_file2)
        assert result == content2


def test_file_read_tool_chunk_reading():
    """Test FileReadTool reading specific chunks of a file."""
    test_file = "/tmp/multiline_test.txt"
    lines = [
        "Line 1\n",
        "Line 2\n",
        "Line 3\n",
        "Line 4\n",
        "Line 5\n",
        "Line 6\n",
        "Line 7\n",
        "Line 8\n",
        "Line 9\n",
        "Line 10\n",
    ]
    file_content = "".join(lines)

    with patch("builtins.open", mock_open(read_data=file_content)):
        tool = FileReadTool()

        # Test reading a specific chunk (lines 3-5)
        result = tool._run(file_path=test_file, start_line=3, line_count=3)
        expected = "".join(lines[2:5])  # Lines are 0-indexed in the array
        assert result == expected

        # Test reading from a specific line to the end
        result = tool._run(file_path=test_file, start_line=8)
        expected = "".join(lines[7:])
        assert result == expected

        # Test with default values (should read entire file)
        result = tool._run(file_path=test_file)
        expected = "".join(lines)
        assert result == expected

        # Test when start_line is 1 but line_count is specified
        result = tool._run(file_path=test_file, start_line=1, line_count=5)
        expected = "".join(lines[0:5])
        assert result == expected


def test_file_read_tool_chunk_error_handling():
    """Test error handling for chunk reading."""
    test_file = "/tmp/short_test.txt"
    lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
    file_content = "".join(lines)

    with patch("builtins.open", mock_open(read_data=file_content)):
        tool = FileReadTool()

        # Test start_line exceeding file length
        result = tool._run(file_path=test_file, start_line=10)
        assert "Error: Start line 10 exceeds the number of lines in the file" in result

        # Test reading partial chunk when line_count exceeds available lines
        result = tool._run(file_path=test_file, start_line=2, line_count=10)
        expected = "".join(lines[1:])  # Should return from line 2 to end
        assert result == expected


def test_file_read_tool_zero_or_negative_start_line():
    """Test that start_line values of 0 or negative read from the start of the file."""
    test_file = "/tmp/negative_test.txt"
    lines = ["Line 1\n", "Line 2\n", "Line 3\n", "Line 4\n", "Line 5\n"]
    file_content = "".join(lines)

    with patch("builtins.open", mock_open(read_data=file_content)):
        tool = FileReadTool()

        # Test with start_line = None
        result = tool._run(file_path=test_file, start_line=None)
        expected = "".join(lines)  # Should read the entire file
        assert result == expected

        # Test with start_line = 0
        result = tool._run(file_path=test_file, start_line=0)
        expected = "".join(lines)  # Should read the entire file
        assert result == expected

        # Test with start_line = 0 and limited line count
        result = tool._run(file_path=test_file, start_line=0, line_count=3)
        expected = "".join(lines[0:3])  # Should read first 3 lines
        assert result == expected

        # Test with negative start_line
        result = tool._run(file_path=test_file, start_line=-5)
        expected = "".join(lines)  # Should read the entire file
        assert result == expected

        # Test with negative start_line and limited line count
        result = tool._run(file_path=test_file, start_line=-10, line_count=2)
        expected = "".join(lines[0:2])  # Should read first 2 lines
        assert result == expected
