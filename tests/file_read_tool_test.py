import os

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
    # Create a temporary test file
    test_file = "/tmp/test_file.txt"
    test_content = "Hello, World!"
    with open(test_file, "w") as f:
        f.write(test_content)

    # Test reading file with runtime file_path
    tool = FileReadTool()
    result = tool._run(file_path=test_file)
    assert result == test_content

    # Clean up
    os.remove(test_file)


def test_file_read_tool_error_handling():
    """Test FileReadTool error handling."""
    # Test missing file path
    tool = FileReadTool()
    result = tool._run()
    assert "Error: No file path provided" in result

    # Test non-existent file
    result = tool._run(file_path="/nonexistent/file.txt")
    assert "Error: File not found at path:" in result

    # Test permission error (create a file without read permissions)
    test_file = "/tmp/no_permission.txt"
    with open(test_file, "w") as f:
        f.write("test")
    os.chmod(test_file, 0o000)

    result = tool._run(file_path=test_file)
    assert "Error: Permission denied" in result

    # Clean up
    os.chmod(test_file, 0o666)  # Restore permissions to delete
    os.remove(test_file)


def test_file_read_tool_constructor_and_run():
    """Test FileReadTool using both constructor and runtime file paths."""
    # Create two test files
    test_file1 = "/tmp/test1.txt"
    test_file2 = "/tmp/test2.txt"
    content1 = "File 1 content"
    content2 = "File 2 content"

    with open(test_file1, "w") as f1, open(test_file2, "w") as f2:
        f1.write(content1)
        f2.write(content2)

    # Test that constructor file_path works
    tool = FileReadTool(file_path=test_file1)
    result = tool._run()
    assert result == content1

    # Test that runtime file_path overrides constructor
    result = tool._run(file_path=test_file2)
    assert result == content2

    # Clean up
    os.remove(test_file1)
    os.remove(test_file2)
