import tempfile
from unittest.mock import patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.pdf_loader import PDFLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


pymupdf = pytest.importorskip("pymupdf")

# Patched at the loader's seam rather than at requests.get: safe_get_bounded
# resolves the hostname before issuing a request, which would make these tests
# depend on DNS for example.com.
FETCH = "crewai_tools.rag.loaders.pdf_loader.safe_get_bounded"


def build_pdf(text: str = "Quarterly revenue was 42") -> bytes:
    """Return the bytes of a one-page PDF containing *text*."""
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    try:
        return document.tobytes()
    finally:
        document.close()


def fetch_result(body: bytes, url: str = "https://example.com/report.pdf"):
    """Build the (body, content_type, final_url) tuple safe_get_bounded returns."""
    return body, "application/pdf", url


class TestPDFLoader:
    def test_load_pdf_from_file(self):
        """A PDF on disk has its text extracted with page markers."""
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
        """A PDF fetched from a URL is extracted and attributed to that URL."""
        with patch(FETCH) as fetch:
            fetch.return_value = fetch_result(build_pdf("Content from URL"))
            result = PDFLoader().load(SourceContent("https://example.com/report.pdf"))

        assert "Content from URL" in result.content
        assert result.source == "https://example.com/report.pdf"
        assert result.metadata["file_name"] == "report.pdf"

        headers = fetch.call_args.kwargs["headers"]
        assert headers["Accept"] == "application/pdf"
        assert "crewai-tools PDFLoader" in headers["User-Agent"]

    def test_load_pdf_from_url_leaves_no_temp_file(self):
        """The URL path must not write a temp file it never cleans up.

        It previously used NamedTemporaryFile(delete=False) without unlinking,
        so every PDF ingested from a URL left a file behind.
        """
        with (
            patch(FETCH) as fetch,
            patch("tempfile.NamedTemporaryFile") as mock_tempfile,
        ):
            fetch.return_value = fetch_result(build_pdf())
            PDFLoader().load(SourceContent("https://example.com/report.pdf"))

        mock_tempfile.assert_not_called()

    def test_load_pdf_from_url_is_size_bounded(self):
        """The download is capped, since the body is held in memory."""
        with patch(FETCH) as fetch:
            fetch.return_value = fetch_result(build_pdf())
            PDFLoader().load(SourceContent("https://example.com/report.pdf"))

        assert fetch.call_args.kwargs["max_bytes"] == 50 * 1024 * 1024

    def test_load_pdf_from_url_accepts_a_custom_size_limit(self):
        """Callers can lower or raise the ceiling per load."""
        with patch(FETCH) as fetch:
            fetch.return_value = fetch_result(build_pdf())
            PDFLoader().load(
                SourceContent("https://example.com/report.pdf"), max_bytes=1024
            )

        assert fetch.call_args.kwargs["max_bytes"] == 1024

    def test_load_pdf_from_url_with_custom_headers(self):
        """Caller-supplied headers replace the loader's defaults."""
        custom_headers = {"Authorization": "Bearer token"}

        with patch(FETCH) as fetch:
            fetch.return_value = fetch_result(build_pdf())
            PDFLoader().load(
                SourceContent("https://example.com/report.pdf"), headers=custom_headers
            )

        assert fetch.call_args.kwargs["headers"] == custom_headers

    def test_load_pdf_url_download_error(self):
        """A failed download surfaces as a ValueError naming the URL."""
        with patch(FETCH, side_effect=Exception("Network error")):
            with pytest.raises(ValueError, match="Failed to download PDF"):
                PDFLoader().load(SourceContent("https://example.com/report.pdf"))

    def test_load_pdf_url_over_size_limit(self):
        """An oversized body is reported rather than partially parsed."""
        with patch(FETCH, side_effect=ValueError("exceeds the 1024 byte limit")):
            with pytest.raises(ValueError, match="Failed to download PDF"):
                PDFLoader().load(SourceContent("https://example.com/huge.pdf"))

    def test_load_pdf_missing_file(self):
        """A missing local path raises FileNotFoundError, not ValueError."""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            PDFLoader().load(SourceContent("/nonexistent/report.pdf"))

    def test_load_corrupt_pdf_raises_value_error(self):
        """Bytes that are not a parseable PDF produce a read error."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(b"%PDF-1.4 not really a pdf")
            f.flush()

            with pytest.raises(ValueError, match="Error reading PDF"):
                PDFLoader().load(SourceContent(f.name))

    def test_pdf_with_no_extractable_text(self):
        """A PDF whose pages hold no text says so instead of returning empty."""
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
        """The same source yields the same doc_id across loads."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(build_pdf())
            f.flush()

            loader = PDFLoader()
            source = SourceContent(f.name)
            assert loader.load(source).doc_id == loader.load(source).doc_id
