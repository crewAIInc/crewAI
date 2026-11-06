from pathlib import Path
from unittest.mock import patch

from crewai_tools import ArxivPaperTool
import pytest


@pytest.mark.parametrize(
    "title", ["A" * 300, "研究" * 100, "Short title"], ids=["long-ascii", "long-unicode", "short"]
)
@pytest.mark.parametrize("use_title_as_filename", [False, True])
def test_title_pdf_filename_fits_filesystem(
    tmp_path: Path, title: str, use_title_as_filename: bool
) -> None:
    paper = {
        "arxiv_id": "2601_00001v1",
        "title": title,
        "summary": "Example summary",
        "authors": ["Example Author"],
        "published_date": "2026-01-01",
        "pdf_url": "https://arxiv.org/pdf/2601.00001v1",
    }

    def save_pdf(url: str, filename: str) -> None:
        Path(filename).write_bytes(b"%PDF-1.4\n")

    tool = ArxivPaperTool(
        download_pdfs=True,
        use_title_as_filename=use_title_as_filename,
        save_dir=str(tmp_path),
    )
    with (
        patch.object(ArxivPaperTool, "fetch_arxiv_data", return_value=[paper]),
        patch("urllib.request.urlretrieve", side_effect=save_pdf),
        patch("time.sleep"),
    ):
        result = tool.run(search_query="agent papers")

    assert not result.startswith("Failed to fetch or download")
    files = list(tmp_path.glob("*.pdf"))
    assert len(files) == 1
    assert len(files[0].name.encode("utf-8")) <= 255
    assert files[0].read_bytes() == b"%PDF-1.4\n"
    if not use_title_as_filename:
        assert files[0].name == "2601_00001v1.pdf"
    elif title == "Short title":
        assert files[0].name == "Short title.pdf"
