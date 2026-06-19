"""Tests for ``crewai.utilities.tool_files``."""

from __future__ import annotations

from crewai_files import File, ImageFile, PDFFile, TextFile

from crewai.utilities.tool_files import extract_files_from_tool_result


def test_extract_files_returns_none_for_string() -> None:
    """Plain strings are not files and should be passed through unchanged."""
    files, message = extract_files_from_tool_result("hello")
    assert files is None
    assert message is None


def test_extract_files_returns_none_for_dict_without_files() -> None:
    """Dicts that don't only contain ``BaseFile`` values are left alone."""
    files, message = extract_files_from_tool_result({"role": "user", "content": "hi"})
    assert files is None
    assert message is None


def test_extract_files_returns_none_for_empty_collections() -> None:
    """Empty collections should not be treated as file containers."""
    assert extract_files_from_tool_result([]) == (None, None)
    assert extract_files_from_tool_result(()) == (None, None)
    assert extract_files_from_tool_result({}) == (None, None)


def test_extract_files_single_file() -> None:
    """A single ``BaseFile`` instance is wrapped into a one-entry dict."""
    text_file = TextFile(source=b"hello", mode="auto")
    files, message = extract_files_from_tool_result(text_file)
    assert files is not None
    assert message is not None
    assert len(files) == 1
    assert next(iter(files.values())) is text_file
    assert "Added 1 file" in message


def test_extract_files_uses_filename_stem_as_key() -> None:
    """The filename stem (without extension) is used as the dict key."""
    text_file = TextFile(
        source=b"hello",
    )
    text_file.source.filename = "report.txt"  # type: ignore[union-attr]
    files, _ = extract_files_from_tool_result(text_file)
    assert files is not None
    assert "report" in files


def test_extract_files_list_of_files() -> None:
    """Lists of ``BaseFile`` instances are extracted into a dict."""
    file_a = TextFile(source=b"a")
    file_b = TextFile(source=b"b")
    files, message = extract_files_from_tool_result([file_a, file_b])
    assert files is not None
    assert message is not None
    assert len(files) == 2
    assert file_a in files.values()
    assert file_b in files.values()
    assert "Added 2 files" in message


def test_extract_files_tuple_of_files() -> None:
    """Tuples of ``BaseFile`` instances are also extracted."""
    file_a = ImageFile(source=b"\x89PNG\r\n\x1a\n")
    files, _ = extract_files_from_tool_result((file_a,))
    assert files is not None
    assert len(files) == 1
    assert file_a in files.values()


def test_extract_files_dict_of_files() -> None:
    """Dicts mapping names to ``BaseFile`` instances are extracted as-is."""
    file_a = TextFile(source=b"a")
    file_b = PDFFile(source=b"%PDF-1.4 content")
    raw = {"notes": file_a, "report": file_b}
    files, message = extract_files_from_tool_result(raw)
    assert files is not None
    assert message is not None
    assert files["notes"] is file_a
    assert files["report"] is file_b


def test_extract_files_mixed_list_returns_none() -> None:
    """Heterogeneous lists with non-files are not treated as file containers."""
    files, message = extract_files_from_tool_result([TextFile(source=b"a"), "string"])
    assert files is None
    assert message is None


def test_extract_files_generic_file_class() -> None:
    """The generic ``File`` class also works as a file return type."""
    generic = File(source=b"plain text")
    files, _ = extract_files_from_tool_result(generic)
    assert files is not None
    assert generic in files.values()


def test_extract_files_with_duplicate_filenames() -> None:
    """When two files share a filename stem the keys are de-duplicated."""
    file_a = TextFile(source=b"a")
    file_b = TextFile(source=b"b")
    file_a.source.filename = "shared.txt"  # type: ignore[union-attr]
    file_b.source.filename = "shared.txt"  # type: ignore[union-attr]
    files, _ = extract_files_from_tool_result([file_a, file_b])
    assert files is not None
    assert len(files) == 2
    keys = list(files)
    assert keys[0] != keys[1]
