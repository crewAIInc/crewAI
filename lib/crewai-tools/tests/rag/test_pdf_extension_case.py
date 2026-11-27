"""PDF type detection must be case-insensitive."""

from pathlib import Path

from crewai_tools.rag.data_types import DataType, DataTypes


def test_pdf_extension_case_insensitive(tmp_path: Path):
    for name in ("a.pdf", "B.PDF", "C.Pdf"):
        path = tmp_path / name
        path.write_bytes(b"%PDF-1.4\n")

        assert DataTypes.from_content(path) == DataType.PDF_FILE
