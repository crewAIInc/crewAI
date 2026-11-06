from pathlib import Path
from unittest.mock import patch

from crewai_files.core.sources import FilePath
import pytest


def test_filename_containing_double_dots_is_allowed(tmp_path):
    path = tmp_path / "data..2024.txt"
    path.write_text("content", encoding="utf-8")

    with patch(
        "crewai_files.core.sources.detect_content_type_from_path",
        return_value="text/plain",
    ):
        file_path = FilePath(path=path)

    assert file_path.path == path
    assert file_path.content_type == "text/plain"


def test_parent_traversal_is_still_rejected():
    with pytest.raises(ValueError, match="Path traversal not allowed"):
        FilePath(path=Path("parent") / ".." / "file.txt")
