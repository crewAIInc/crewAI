from collections.abc import Callable
import tempfile
from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.docx_loader import DOCXLoader
from crewai_tools.rag.source_content import SourceContent
from docx import Document as RealDocument
import pytest


def build_docx(build: Callable[..., None]) -> object:
    """Return an in-memory python-docx Document populated by ``build``."""
    doc = RealDocument()
    build(doc)
    return doc


@pytest.fixture
def save_docx(tmp_path):
    """Save a document built by ``build`` and return its path."""

    def _save(build: Callable[..., None], name: str = "doc.docx") -> str:
        path = tmp_path / name
        build_docx(build).save(path)
        return str(path)

    return _save


def _paragraphs(*texts: str) -> Callable[..., None]:
    def build(doc) -> None:
        for text in texts:
            doc.add_paragraph(text)

    return build


def _table(doc, rows: list[list[str]]):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            table.cell(r, c).text = text
    return table


class TestDOCXLoader:
    def test_load_docx_from_file(self, save_docx):
        path = save_docx(_paragraphs("First paragraph", "Second paragraph", "   "))

        result = DOCXLoader().load(SourceContent(path))

        assert isinstance(result, LoaderResult)
        assert result.content == "First paragraph\nSecond paragraph"
        assert result.metadata == {"format": "docx", "paragraphs": 3, "tables": 0}
        assert result.source == path

    def test_load_docx_with_tables(self, save_docx):
        def build(doc) -> None:
            doc.add_paragraph("Document with table")
            _table(doc, [["a", "b"]])
            _table(doc, [["c", "d"]])

        result = DOCXLoader().load(SourceContent(save_docx(build)))

        assert result.metadata["tables"] == 2

    @patch("crewai_tools.security.safe_requests._raw_get")
    @patch("docx.Document")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_load_docx_from_url(
        self, mock_unlink, mock_tempfile, mock_docx_class, mock_get
    ):
        mock_get.return_value = Mock(
            content=b"fake docx content", raise_for_status=Mock()
        )

        mock_temp = Mock(name="/tmp/temp_docx_file.docx")
        mock_temp.__enter__ = Mock(return_value=mock_temp)
        mock_temp.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp

        mock_docx_class.return_value = build_docx(_paragraphs("Content from URL"))

        loader = DOCXLoader()
        result = loader.load(SourceContent("https://example.com/test.docx"))

        assert "Content from URL" in result.content
        assert result.source == "https://example.com/test.docx"

        headers = mock_get.call_args[1]["headers"]
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in headers["Accept"]
        )
        assert "crewai-tools DOCXLoader" in headers["User-Agent"]

        mock_temp.write.assert_called_once_with(b"fake docx content")

    @patch("crewai_tools.security.safe_requests._raw_get")
    @patch("docx.Document")
    def test_load_docx_from_url_with_custom_headers(self, mock_docx_class, mock_get):
        mock_get.return_value = Mock(
            content=b"fake docx content", raise_for_status=Mock()
        )
        mock_docx_class.return_value = build_docx(lambda doc: None)

        loader = DOCXLoader()
        custom_headers = {"Authorization": "Bearer token"}

        with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
            loader.load(
                SourceContent("https://example.com/test.docx"), headers=custom_headers
            )

        assert mock_get.call_args[1]["headers"] == custom_headers

    @patch("crewai_tools.security.safe_requests._raw_get")
    def test_load_docx_url_download_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        loader = DOCXLoader()
        with pytest.raises(ValueError, match="Error fetching content from URL"):
            loader.load(SourceContent("https://example.com/test.docx"))

    @patch("crewai_tools.security.safe_requests._raw_get")
    def test_load_docx_url_http_error(self, mock_get):
        mock_get.return_value = Mock(
            raise_for_status=Mock(side_effect=Exception("404 Not Found"))
        )

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

    def test_load_docx_empty_document(self, save_docx):
        result = DOCXLoader().load(SourceContent(save_docx(lambda doc: None)))

        assert result.content == ""
        assert result.metadata == {"paragraphs": 0, "tables": 0, "format": "docx"}

    def test_docx_doc_id_generation(self, save_docx):
        source = SourceContent(save_docx(_paragraphs("Consistent content")))

        loader = DOCXLoader()
        assert loader.load(source).doc_id == loader.load(source).doc_id


class TestDOCXLoaderTables:
    """Table text must be extracted, in document order (#7846)."""

    def test_table_only_document_is_not_empty(self, save_docx):
        path = save_docx(lambda doc: _table(doc, [["Example item", "42 EUR"]]))

        result = DOCXLoader().load(SourceContent(path))

        assert result.content == "Example item | 42 EUR"
        assert result.metadata == {"format": "docx", "paragraphs": 0, "tables": 1}

    def test_paragraphs_and_tables_keep_document_order(self, save_docx):
        def build(doc) -> None:
            doc.add_paragraph("Price list")
            _table(doc, [["Item", "Price"], ["Example item", "42 EUR"]])
            doc.add_paragraph("Thanks for your order.")

        result = DOCXLoader().load(SourceContent(save_docx(build)))

        assert result.content == (
            "Price list\nItem | Price\nExample item | 42 EUR\nThanks for your order."
        )

    def test_nested_table_text_is_included(self, save_docx):
        def build(doc) -> None:
            outer = _table(doc, [["Region", "Details"]])
            cell = outer.cell(0, 1)
            cell.text = "Breakdown:"
            inner = cell.add_table(rows=1, cols=2)
            inner.cell(0, 0).text = "Q1"
            inner.cell(0, 1).text = "10"

        result = DOCXLoader().load(SourceContent(save_docx(build)))

        assert result.content == "Region | Breakdown: Q1 | 10"

    def test_horizontally_merged_cell_is_not_duplicated(self, save_docx):
        def build(doc) -> None:
            table = _table(doc, [["", "", "Total"], ["a", "b", "3"]])
            merged = table.cell(0, 0).merge(table.cell(0, 1))
            merged.text = "Spanning header"

        result = DOCXLoader().load(SourceContent(save_docx(build)))

        assert result.content == "Spanning header | Total\na | b | 3"

    def test_empty_rows_are_skipped_and_empty_cells_kept_in_place(self, save_docx):
        def build(doc) -> None:
            _table(doc, [["Name", "", "Note"], ["", "", ""], ["x", "", "y"]])

        result = DOCXLoader().load(SourceContent(save_docx(build)))

        assert result.content == "Name |  | Note\nx |  | y"

    def test_unicode_table_text(self, save_docx):
        path = save_docx(lambda doc: _table(doc, [["東京", "¥4,200"], ["Café", "🙂"]]))

        result = DOCXLoader().load(SourceContent(path))

        assert result.content == "東京 | ¥4,200\nCafé | 🙂"

    def test_document_without_tables_is_unchanged(self, save_docx):
        build = _paragraphs("One", "", "Two", "  ", "Three")

        result = DOCXLoader().load(SourceContent(save_docx(build)))

        assert result.content == "One\nTwo\nThree"
        assert result.metadata == {"format": "docx", "paragraphs": 5, "tables": 0}

    @patch("crewai_tools.security.safe_requests._raw_get")
    def test_table_from_url(self, mock_get, tmp_path):
        source = tmp_path / "remote.docx"

        def build(doc) -> None:
            doc.add_paragraph("Remote prices")
            _table(doc, [["Example item", "42 EUR"]])

        build_docx(build).save(source)
        mock_get.return_value = Mock(
            content=source.read_bytes(), raise_for_status=Mock()
        )

        result = DOCXLoader().load(SourceContent("https://example.com/prices.docx"))

        assert result.content == "Remote prices\nExample item | 42 EUR"
        assert result.metadata["tables"] == 1

    def test_large_table(self, save_docx):
        rows = [[f"r{r}c{c}" for c in range(6)] for r in range(200)]
        result = DOCXLoader().load(
            SourceContent(save_docx(lambda doc: _table(doc, rows)))
        )

        lines = result.content.split("\n")
        assert len(lines) == 200
        assert lines[0] == "r0c0 | r0c1 | r0c2 | r0c3 | r0c4 | r0c5"
        assert lines[-1] == "r199c0 | r199c1 | r199c2 | r199c3 | r199c4 | r199c5"
