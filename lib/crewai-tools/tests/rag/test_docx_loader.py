import tempfile
from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.docx_loader import (
    DEFAULT_MAX_DOCX_BYTES,
    DOCXLoader,
)
from crewai_tools.rag.source_content import SourceContent
import pytest


class TestDOCXLoader:
    @patch("docx.Document")
    def test_load_docx_from_file(self, mock_docx_class):
        mock_doc = Mock()
        mock_doc.paragraphs = [
            Mock(text="First paragraph"),
            Mock(text="Second paragraph"),
            Mock(text="   "),
        ]
        mock_doc.tables = []
        mock_docx_class.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx") as f:
            loader = DOCXLoader()
            result = loader.load(SourceContent(f.name))

            assert isinstance(result, LoaderResult)
            assert result.content == "First paragraph\nSecond paragraph"
            assert result.metadata == {"format": "docx", "paragraphs": 3, "tables": 0}
            assert result.source == f.name

    @patch("docx.Document")
    def test_load_docx_with_tables(self, mock_docx_class):
        mock_doc = Mock()
        mock_doc.paragraphs = [Mock(text="Document with table")]
        mock_doc.tables = [Mock(), Mock()]
        mock_docx_class.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".docx") as f:
            loader = DOCXLoader()
            result = loader.load(SourceContent(f.name))

            assert result.metadata["tables"] == 2

    @patch("crewai_tools.rag.loaders.docx_loader.safe_get_bounded")
    @patch("docx.Document")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_load_docx_from_url(
        self, mock_unlink, mock_tempfile, mock_docx_class, mock_get_bounded
    ):
        mock_get_bounded.return_value = (
            b"fake docx content",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "https://example.com/test.docx",
        )

        mock_temp = Mock(name="/tmp/temp_docx_file.docx")
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp

        mock_doc = Mock()
        mock_doc.paragraphs = [Mock(text="Content from URL")]
        mock_doc.tables = []
        mock_docx_class.return_value = mock_doc

        loader = DOCXLoader()
        result = loader.load(SourceContent("https://example.com/test.docx"))

        assert "Content from URL" in result.content
        assert result.source == "https://example.com/test.docx"

        headers = mock_get_bounded.call_args.kwargs["headers"]
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in headers["Accept"]
        )
        assert "crewai-tools DOCXLoader" in headers["User-Agent"]

        assert mock_get_bounded.call_args.kwargs["max_bytes"] == DEFAULT_MAX_DOCX_BYTES
        mock_temp.write.assert_called_once_with(b"fake docx content")
        mock_unlink.assert_called_once_with(mock_temp.name)

    @patch("crewai_tools.rag.loaders.docx_loader.os.unlink")
    @patch("crewai_tools.rag.loaders.docx_loader.safe_get_bounded")
    @patch("docx.Document")
    def test_load_docx_from_url_cleans_up_after_parse_error(
        self, mock_docx_class, mock_get_bounded, mock_unlink
    ):
        mock_get_bounded.return_value = (
            b"invalid docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "https://example.com/test.docx",
        )
        mock_docx_class.side_effect = Exception("Invalid DOCX file")

        loader = DOCXLoader()
        with pytest.raises(ValueError, match="Error loading DOCX file"):
            loader.load(SourceContent("https://example.com/test.docx"))

        mock_unlink.assert_called_once()

    @patch("crewai_tools.rag.loaders.docx_loader.safe_get_bounded")
    @patch("docx.Document")
    def test_load_docx_from_url_with_custom_options(
        self, mock_docx_class, mock_get_bounded
    ):
        mock_get_bounded.return_value = (
            b"fake docx content",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "https://example.com/test.docx",
        )
        mock_docx_class.return_value = Mock(paragraphs=[], tables=[])

        loader = DOCXLoader()
        custom_headers = {"Authorization": "Bearer token"}

        with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
            loader.load(
                SourceContent("https://example.com/test.docx"),
                headers=custom_headers,
                max_bytes=1024,
            )

        assert mock_get_bounded.call_args.kwargs["headers"] == custom_headers
        assert mock_get_bounded.call_args.kwargs["max_bytes"] == 1024

    @patch("crewai_tools.rag.loaders.docx_loader.tempfile.NamedTemporaryFile")
    @patch("crewai_tools.rag.loaders.docx_loader.safe_get_bounded")
    def test_load_docx_url_rejects_oversized_response(
        self, mock_get_bounded, mock_tempfile
    ):
        mock_get_bounded.side_effect = ValueError(
            "Response body exceeds the 52428800 byte limit."
        )

        loader = DOCXLoader()
        with pytest.raises(ValueError, match="exceeds the 52428800 byte limit"):
            loader.load(SourceContent("https://example.com/large.docx"))

        mock_tempfile.assert_not_called()

    @patch("crewai_tools.rag.loaders.docx_loader.safe_get_bounded")
    def test_load_docx_url_download_error(self, mock_get_bounded):
        mock_get_bounded.side_effect = Exception("Network error")

        loader = DOCXLoader()
        with pytest.raises(ValueError, match="Error fetching content from URL"):
            loader.load(SourceContent("https://example.com/test.docx"))

    @patch("crewai_tools.rag.loaders.docx_loader.safe_get_bounded")
    def test_load_docx_url_http_error(self, mock_get_bounded):
        mock_get_bounded.side_effect = Exception("404 Not Found")

        loader = DOCXLoader()
        with pytest.raises(ValueError, match="Error fetching content from URL"):
            loader.load(SourceContent("https://example.com/notfound.docx"))

    def test_load_docx_invalid_source(self):
        loader = DOCXLoader()
        with pytest.raises(ValueError, match="Source must be a valid file path or URL"):
            loader.load(SourceContent("not_a_file_or_url"))

    @patch("docx.Document")
    def test_load_docx_parsing_error(self, mock_docx_class):
        mock_docx_class.side_effect = Exception("Invalid DOCX file")

        with tempfile.NamedTemporaryFile(suffix=".docx") as f:
            loader = DOCXLoader()
            with pytest.raises(ValueError, match="Error loading DOCX file"):
                loader.load(SourceContent(f.name))

    @patch("docx.Document")
    def test_load_docx_empty_document(self, mock_docx_class):
        mock_docx_class.return_value = Mock(paragraphs=[], tables=[])

        with tempfile.NamedTemporaryFile(suffix=".docx") as f:
            loader = DOCXLoader()
            result = loader.load(SourceContent(f.name))

            assert result.content == ""
            assert result.metadata == {"paragraphs": 0, "tables": 0, "format": "docx"}

    @patch("docx.Document")
    def test_docx_doc_id_generation(self, mock_docx_class):
        mock_docx_class.return_value = Mock(
            paragraphs=[Mock(text="Consistent content")], tables=[]
        )

        with tempfile.NamedTemporaryFile(suffix=".docx") as f:
            loader = DOCXLoader()
            source = SourceContent(f.name)
            assert loader.load(source).doc_id == loader.load(source).doc_id
