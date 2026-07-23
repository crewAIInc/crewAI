import os
import tempfile
from collections.abc import Callable, Iterator

from crewai_tools.rag.data_types import DataType, DataTypes
import pytest


@pytest.fixture
def temp_file() -> Iterator[Callable[[str], str]]:
    created_files: list[str] = []

    def _create(suffix: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
        f.write("content")
        f.close()
        created_files.append(f.name)
        return f.name

    yield _create

    for path in created_files:
        os.unlink(path)


class TestDataTypesFromContent:
    @pytest.mark.parametrize(
        "suffix,expected",
        [
            (".pdf", DataType.PDF_FILE),
            (".PDF", DataType.PDF_FILE),
            (".csv", DataType.CSV),
            (".CSV", DataType.CSV),
            (".docx", DataType.DOCX),
            (".DOCX", DataType.DOCX),
            (".json", DataType.JSON),
            (".Json", DataType.JSON),
            (".xml", DataType.XML),
            (".XML", DataType.XML),
            (".md", DataType.MDX),
            (".MD", DataType.MDX),
        ],
    )
    def test_file_extension_detection_is_case_insensitive(
        self, temp_file: Callable[[str], str], suffix: str, expected: DataType
    ) -> None:
        path = temp_file(suffix)
        assert DataTypes.from_content(path) == expected

    def test_uppercase_pdf_url_is_detected_as_pdf(self) -> None:
        assert (
            DataTypes.from_content("https://example.com/file.PDF")
            == DataType.PDF_FILE
        )

    def test_lowercase_pdf_url_is_detected_as_pdf(self) -> None:
        assert (
            DataTypes.from_content("https://example.com/file.pdf")
            == DataType.PDF_FILE
        )
