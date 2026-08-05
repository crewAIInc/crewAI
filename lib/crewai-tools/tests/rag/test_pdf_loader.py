import tempfile
from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.pdf_loader import PDFLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


pymupdf = pytest.importorskip("pymupdf")


def build_pdf(text: str = "Quarterly revenue was 42") -> bytes:
    """Return the bytes of a one-page PDF containing *text*."""
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    try:
        return document.tobytes()
    finally:
        document.close()


class TestPDFLoader:
    def test_load_pdf_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(build_pdf())
            f.flush()

            result = PDFLoader().load(SourceContent(f.name))

        assert isinstance(result, LoaderResult)
        assert "Page 1:" in result.content
        assert "Quarterly revenue was 42" in result.content
        assert result.metadata["num_pages"] == 1
        assert result.metadata["file_type"] == "pdf"

    def test_load_pdf_from_url(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(
                content=build_pdf("Content from URL"),
                raise_for_status=Mock(),
                status_code=200,
                headers={},
            )
            result = PDFLoader().load(SourceContent("https://example.com/report.pdf"))

        assert "Content from URL" in result.content
        assert result.source == "https://example.com/report.pdf"
        assert result.metadata["file_name"] == "report.pdf"

        headers = mock_get.call_args[1]["headers"]
        assert headers["Accept"] == "application/pdf"
        assert "crewai-tools PDFLoader" in headers["User-Agent"]

    def test_load_pdf_from_url_leaves_no_temp_file(self):
        """The URL path must not write a temp file it never cleans up.

        It previously used NamedTemporaryFile(delete=False) without unlinking,
        so every PDF ingested from a URL left a file behind.
        """
        with (
            patch("requests.get") as mock_get,
            patch("tempfile.NamedTemporaryFile") as mock_tempfile,
        ):
            mock_get.return_value = Mock(
                content=build_pdf(),
                raise_for_status=Mock(),
                status_code=200,
                headers={},
            )
            PDFLoader().load(SourceContent("https://example.com/report.pdf"))

        mock_tempfile.assert_not_called()

    def test_load_pdf_from_url_with_custom_headers(self):
        custom_headers = {"Authorization": "Bearer token"}

        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(
                content=build_pdf(),
                raise_for_status=Mock(),
                status_code=200,
                headers={},
            )
            PDFLoader().load(
                SourceContent("https://example.com/report.pdf"), headers=custom_headers
            )

        assert mock_get.call_args[1]["headers"] == custom_headers

    def test_load_pdf_url_download_error(self):
        with patch("requests.get", side_effect=Exception("Network error")):
            with pytest.raises(ValueError, match="Failed to download PDF"):
                PDFLoader().load(SourceContent("https://example.com/report.pdf"))

    def test_load_pdf_missing_file(self):
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            PDFLoader().load(SourceContent("/nonexistent/report.pdf"))

    def test_load_corrupt_pdf_raises_value_error(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(b"%PDF-1.4 not really a pdf")
            f.flush()

            with pytest.raises(ValueError, match="Error reading PDF"):
                PDFLoader().load(SourceContent(f.name))

    def test_pdf_with_no_extractable_text(self):
        document = pymupdf.open()
        document.new_page()
        blank = document.tobytes()
        document.close()

        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(blank)
            f.flush()

            result = PDFLoader().load(SourceContent(f.name))

        assert "no extractable text" in result.content

    def test_pdf_doc_id_is_stable(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(build_pdf())
            f.flush()

            loader = PDFLoader()
            source = SourceContent(f.name)
            assert loader.load(source).doc_id == loader.load(source).doc_id
