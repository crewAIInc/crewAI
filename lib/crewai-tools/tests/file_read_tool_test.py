from unittest.mock import mock_open, patch

from crewai_tools import FileReadTool


class CountingFile:
    """Text-file stand-in that records how many lines were actually consumed."""

    def __init__(self, lines):
        self._lines = lines
        self.consumed = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed >= len(self._lines):
            raise StopIteration
        line = self._lines[self.consumed]
        self.consumed += 1
        return line

    def read(self):
        self.consumed = len(self._lines)
        return "".join(self._lines)


def test_file_read_tool_constructor():
    """Test FileReadTool initialization with file_path."""
    test_file = "test_file.txt"

    tool = FileReadTool(file_path=test_file)
    assert tool.file_path == test_file
    assert "test_file.txt" in tool.description


def test_file_read_tool_run():
    """Test FileReadTool _run method with file_path at runtime."""
    test_file = "test_file.txt"
    test_content = "Hello, World!"

    # Use mock_open to mock file operations
    with patch("builtins.open", mock_open(read_data=test_content)):
        tool = FileReadTool()
        result = tool._run(file_path=test_file)
        assert result == test_content


def test_file_read_tool_error_handling():
    """Test FileReadTool error handling."""
    tool = FileReadTool()
    result = tool._run()
    assert "Error: No file path provided" in result

    result = tool._run(file_path="nonexistent/file.txt")
    assert "Error: File not found at path:" in result

    with patch("builtins.open", side_effect=PermissionError()):
        result = tool._run(file_path="no_permission.txt")
        assert "Error: Permission denied" in result


def test_file_read_tool_constructor_and_run():
    """Test FileReadTool using both constructor and runtime file paths."""
    test_file1 = "test1.txt"
    test_file2 = "test2.txt"
    content1 = "File 1 content"
    content2 = "File 2 content"

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
    test_file = "multiline_test.txt"
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
    test_file = "short_test.txt"
    lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
    file_content = "".join(lines)

    with patch("builtins.open", mock_open(read_data=file_content)):
        tool = FileReadTool()

        result = tool._run(file_path=test_file, start_line=10)
        assert "Error: Start line 10 exceeds the number of lines in the file" in result

        # Test reading partial chunk when line_count exceeds available lines
        result = tool._run(file_path=test_file, start_line=2, line_count=10)
        expected = "".join(lines[1:])  # Should return from line 2 to end
        assert result == expected


def test_file_read_tool_zero_or_negative_start_line():
    """Test that start_line values of 0 or negative read from the start of the file."""
    test_file = "negative_test.txt"
    lines = ["Line 1\n", "Line 2\n", "Line 3\n", "Line 4\n", "Line 5\n"]
    file_content = "".join(lines)

    with patch("builtins.open", mock_open(read_data=file_content)):
        tool = FileReadTool()

        result = tool._run(file_path=test_file, start_line=None)
        expected = "".join(lines)  # Should read the entire file
        assert result == expected

        result = tool._run(file_path=test_file, start_line=0)
        expected = "".join(lines)  # Should read the entire file
        assert result == expected

        # Test with start_line = 0 and limited line count
        result = tool._run(file_path=test_file, start_line=0, line_count=3)
        expected = "".join(lines[0:3])  # Should read first 3 lines
        assert result == expected

        result = tool._run(file_path=test_file, start_line=-5)
        expected = "".join(lines)  # Should read the entire file
        assert result == expected

        # Test with negative start_line and limited line count
        result = tool._run(file_path=test_file, start_line=-10, line_count=2)
        expected = "".join(lines[0:2])  # Should read first 2 lines
        assert result == expected


def test_file_read_tool_error_messages_do_not_disclose_absolute_paths(
    tmp_path, monkeypatch
):
    """FileReadTool should redact absolute prefixes from user-visible errors."""
    monkeypatch.chdir(tmp_path)
    tool = FileReadTool()
    target = tmp_path / "secret.txt"

    result = tool._run(file_path=str(target))
    assert "secret.txt" in result
    assert str(tmp_path) not in result

    target.touch()
    with patch("builtins.open", side_effect=PermissionError()):
        result = tool._run(file_path=str(target))
    assert "secret.txt" in result
    assert str(tmp_path) not in result

    with patch(
        "builtins.open",
        side_effect=OSError(5, "Input/output error", str(target)),
    ):
        result = tool._run(file_path=str(target))
    assert "secret.txt" in result
    assert str(tmp_path) not in result


def test_file_read_tool_invalid_path_error_does_not_disclose_workspace(
    tmp_path, monkeypatch
):
    """Validation errors should not echo the resolved workspace path."""
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.txt"

    result = FileReadTool()._run(file_path=str(outside))

    assert "Invalid file path" in result
    assert "outside.txt" in result
    assert str(tmp_path) not in result
    assert str(tmp_path.parent) not in result
    # Point users at base_dir, not the process-wide escape hatch.
    assert "base_dir" in result
    assert "CREWAI_TOOLS_ALLOW_UNSAFE_PATHS" not in result


def test_constructor_path_outside_working_directory_is_readable(tmp_path, monkeypatch):
    """A developer-declared file_path is trusted even outside the sandbox."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    target = tmp_path / "declared.txt"
    target.write_text("declared content")

    tool = FileReadTool(file_path=str(target))

    assert tool._run() == "declared content"
    assert tool._run(file_path=str(target)) == "declared content"


def test_declared_file_is_reachable_the_way_an_agent_calls_it(tmp_path, monkeypatch):
    """The label in the description must resolve to the declared file.

    The description only shows a redacted label, so that label is all the model
    has to work with. It has to address the declared file.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    target = tmp_path / "declared.txt"
    target.write_text("declared content")

    tool = FileReadTool(file_path=str(target))
    label = tool._declared_label

    assert label == "declared.txt"
    assert label in tool.description
    # Omitting file_path entirely, and passing the advertised label, both work.
    assert tool.run() == "declared content"
    assert tool.run(file_path=label) == "declared content"


def test_run_without_file_path_reports_error_when_no_default(tmp_path, monkeypatch):
    """file_path is optional in the schema, so this must not raise."""
    monkeypatch.chdir(tmp_path)

    assert "Error: No file path provided" in FileReadTool().run()


def test_declared_relative_path_survives_chdir(tmp_path, monkeypatch):
    """The declared file is pinned at construction, not re-resolved per call."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel.txt").write_text("original")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "rel.txt").write_text("a different file")

    tool = FileReadTool(file_path="rel.txt")
    assert tool._run() == "original"

    monkeypatch.chdir(nested)
    assert tool._run() == "original"
    assert tool._run(file_path="rel.txt") == "original"


def test_relative_declared_path_anchors_to_base_dir(tmp_path, monkeypatch):
    """A relative declared path must resolve against base_dir, not the cwd.

    The advertised label is built against base_dir, so pinning against the cwd
    would make the same name mean two different files — and would serve a file
    from outside base_dir under a label that looks like it is inside.
    """
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (allowed / "data.txt").write_text("sandbox file")
    (work / "data.txt").write_text("cwd file")
    monkeypatch.chdir(work)

    tool = FileReadTool(file_path="data.txt", base_dir=str(allowed))

    assert tool._declared_label == "data.txt"
    assert tool._declared_realpath == str(allowed / "data.txt")
    assert tool.run() == "sandbox file"
    assert tool.run(file_path="data.txt") == "sandbox file"


def test_relative_base_dir_is_anchored_at_construction(tmp_path, monkeypatch):
    """A relative base_dir must not follow a later chdir.

    Otherwise the sandbox root moves while the declared file stays pinned, and
    the tool applies two different roots.
    """
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "data.txt").write_text("sandbox file")
    nested = tmp_path / "sub"
    nested.mkdir()

    tool = FileReadTool(base_dir="allowed")
    assert tool.base_dir == str(allowed)

    monkeypatch.chdir(nested)
    assert tool._run(file_path="data.txt") == "sandbox file"


def test_declared_path_survives_a_serialization_round_trip(tmp_path, monkeypatch):
    """model_dump drops private attrs, so the pin must be rebuilt on restore."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "data.txt").write_text("sandbox file")

    tool = FileReadTool(file_path="data.txt", base_dir=str(allowed))
    restored = FileReadTool.model_validate(tool.model_dump())

    assert restored._declared_realpath == tool._declared_realpath
    assert restored.run() == "sandbox file"

    # A chdir between dump and restore must not repoint the declared file.
    nested = workspace / "sub"
    nested.mkdir()
    monkeypatch.chdir(nested)
    assert FileReadTool.model_validate(tool.model_dump()).run() == "sandbox file"


def test_constructor_path_does_not_widen_the_sandbox(tmp_path, monkeypatch):
    """Declaring one file must not expose its siblings to the LLM."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    declared = tmp_path / "declared.txt"
    declared.write_text("declared content")
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("secret")

    result = FileReadTool(file_path=str(declared))._run(file_path=str(sibling))

    assert "Invalid file path" in result
    assert "secret" not in result


def test_base_dir_widens_the_sandbox(tmp_path, monkeypatch):
    """base_dir lets a developer authorize reads outside the working directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    data = tmp_path / "data"
    data.mkdir()
    (data / "report.csv").write_text("a,b,c")

    tool = FileReadTool(base_dir=str(data))

    assert tool._run(file_path=str(data / "report.csv")) == "a,b,c"
    assert tool._run(file_path="report.csv") == "a,b,c"


def test_base_dir_still_blocks_escapes(tmp_path, monkeypatch):
    """base_dir moves the sandbox; it does not remove it."""
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    result = FileReadTool(base_dir=str(data))._run(file_path=str(outside))

    assert "Invalid file path" in result
    assert "secret" not in result


def test_line_window_stops_reading_early():
    """A small window must not scan the rest of the file."""
    handle = CountingFile([f"Line {i}\n" for i in range(1, 1001)])

    with patch("builtins.open", return_value=handle):
        result = FileReadTool()._run(
            file_path="big.log", start_line=1, line_count=3
        )

    assert result == "Line 1\nLine 2\nLine 3\n"
    assert handle.consumed == 3


def test_line_window_stops_early_with_offset():
    handle = CountingFile([f"Line {i}\n" for i in range(1, 1001)])

    with patch("builtins.open", return_value=handle):
        result = FileReadTool()._run(
            file_path="big.log", start_line=10, line_count=2
        )

    assert result == "Line 10\nLine 11\n"
    assert handle.consumed == 11


def test_reads_utf8_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    content = "café — 日本語 — 🚀"
    (tmp_path / "unicode.txt").write_bytes(content.encode("utf-8"))

    assert FileReadTool()._run(file_path="unicode.txt") == content


def test_encoding_is_configurable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "latin.txt").write_bytes("café".encode("latin-1"))

    assert FileReadTool(encoding="latin-1")._run(file_path="latin.txt") == "café"


def test_null_byte_path_returns_error_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = FileReadTool()._run(file_path="a\x00b.txt")

    assert "Error" in result


def test_decode_error_names_the_encoding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "binary.bin").write_bytes(bytes(range(256)))

    result = FileReadTool()._run(file_path="binary.bin")

    assert "Failed to decode" in result
    assert "utf-8" in result
    assert str(tmp_path) not in result
