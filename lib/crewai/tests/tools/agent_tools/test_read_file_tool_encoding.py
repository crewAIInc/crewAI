"""Encoding fallback tests for ReadFileTool."""

from crewai.tools.agent_tools.read_file_tool import ReadFileTool


class _FakeTextFile:
    """Duck-typed stand-in for a text file input.

    The real ``crewai_files.TextFile`` detects its content type with libmagic,
    which is not available in every environment, so this test keeps to the
    attributes ``ReadFileTool`` reads.
    """

    def __init__(self, content: bytes, filename: str = "notes.txt") -> None:
        self._content = content
        self.content_type = "text/plain"
        self.filename = filename

    def read(self) -> bytes:
        return self._content


def test_non_utf8_text_file_is_decoded_with_replacement() -> None:
    """A text file in another encoding must be readable, not raise."""
    tool = ReadFileTool()
    tool.set_files({"notes.txt": _FakeTextFile("café".encode("latin-1"))})

    result = tool._run(file_name="notes.txt")

    assert result.startswith("caf")
    assert "�" in result
