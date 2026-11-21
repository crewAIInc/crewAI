from __future__ import annotations

from crewai_files import TextFile
from crewai_files.core.sources import FileBytes
from crewai_files.resolution.utils import normalize_input_files


def test_normalize_base_file_uses_full_filename() -> None:
    result = normalize_input_files(
        [TextFile(source=FileBytes(data=b"content", filename="notes.txt"))]
    )

    assert list(result) == ["notes.txt"]


def test_normalize_base_files_with_same_stem_keeps_both() -> None:
    result = normalize_input_files(
        [
            TextFile(source=FileBytes(data=b"json", filename="data.json")),
            TextFile(source=FileBytes(data=b"text", filename="data.txt")),
        ]
    )

    assert set(result) == {"data.json", "data.txt"}
    assert result["data.json"].read() == b"json"
    assert result["data.txt"].read() == b"text"
