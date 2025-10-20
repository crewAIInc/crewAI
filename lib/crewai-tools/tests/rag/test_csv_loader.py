import os
import tempfile
from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.csv_loader import CSVLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


@pytest.fixture
def temp_csv_file():
    created_files = []

    def _create(content: str):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(content)
        f.close()
        created_files.append(f.name)
        return f.name

    yield _create

    for path in created_files:
        os.unlink(path)


class TestCSVLoader:
    def test_load_csv_from_file(self, temp_csv_file):
        path = temp_csv_file("name,age,city\nJohn,25,New York\nJane,30,Chicago")
        loader = CSVLoader()
        result = loader.load(SourceContent(path))

        assert isinstance(result, LoaderResult)
        assert "Headers: name | age | city" in result.content
        assert "Row 1: name: John | age: 25 | city: New York" in result.content
        assert "Row 2: name: Jane | age: 30 | city: Chicago" in result.content
        assert result.metadata == {
            "format": "csv",
            "columns": ["name", "age", "city"],
            "rows": 2,
        }
        assert result.source == path
        assert result.doc_id

    def test_load_csv_with_empty_values(self, temp_csv_file):
        path = temp_csv_file("name,age,city\nJohn,,New York\n,30,")
        result = CSVLoader().load(SourceContent(path))

        assert "Row 1: name: John | city: New York" in result.content
        assert "Row 2: age: 30" in result.content
        assert result.metadata["rows"] == 2

    def test_load_csv_malformed(self, temp_csv_file):
        path = temp_csv_file('invalid,csv\nunclosed quote "missing')
        result = CSVLoader().load(SourceContent(path))

        assert "Headers: invalid | csv" in result.content
        assert 'Row 1: invalid: unclosed quote "missing' in result.content
        assert result.metadata["columns"] == ["invalid", "csv"]

    def test_load_csv_empty_file(self, temp_csv_file):
        path = temp_csv_file("")
        result = CSVLoader().load(SourceContent(path))

        assert result.content == ""
        assert result.metadata["rows"] == 0

    def test_load_csv_text_input(self):
        raw_csv = "col1,col2\nvalue1,value2\nvalue3,value4"
        result = CSVLoader().load(SourceContent(raw_csv))

        assert "Headers: col1 | col2" in result.content
        assert "Row 1: col1: value1 | col2: value2" in result.content
        assert "Row 2: col1: value3 | col2: value4" in result.content
        assert result.metadata["columns"] == ["col1", "col2"]
        assert result.metadata["rows"] == 2

    def test_doc_id_is_deterministic(self, temp_csv_file):
        path = temp_csv_file("name,value\ntest,123")
        loader = CSVLoader()

        result1 = loader.load(SourceContent(path))
        result2 = loader.load(SourceContent(path))

        assert result1.doc_id == result2.doc_id

    @patch("requests.get")
    def test_load_csv_from_url(self, mock_get):
        mock_get.return_value = Mock(
            text="name,value\ntest,123", raise_for_status=Mock(return_value=None)
        )

        result = CSVLoader().load(SourceContent("https://example.com/data.csv"))

        assert "Headers: name | value" in result.content
        assert "Row 1: name: test | value: 123" in result.content
        headers = mock_get.call_args[1]["headers"]
        assert "text/csv" in headers["Accept"]
        assert "crewai-tools CSVLoader" in headers["User-Agent"]

    @patch("requests.get")
    def test_load_csv_with_custom_headers(self, mock_get):
        mock_get.return_value = Mock(
            text="data,value\ntest,456", raise_for_status=Mock(return_value=None)
        )
        headers = {"Authorization": "Bearer token", "Custom-Header": "value"}
        result = CSVLoader().load(
            SourceContent("https://example.com/data.csv"), headers=headers
        )

        assert "Headers: data | value" in result.content
        assert mock_get.call_args[1]["headers"] == headers

    @patch("requests.get")
    def test_csv_loader_handles_network_errors(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        loader = CSVLoader()

        with pytest.raises(ValueError, match="Error fetching content from URL"):
            loader.load(SourceContent("https://example.com/data.csv"))

    @patch("requests.get")
    def test_csv_loader_handles_http_error(self, mock_get):
        mock_get.return_value = Mock()
        mock_get.return_value.raise_for_status.side_effect = Exception("404 Not Found")
        loader = CSVLoader()

        with pytest.raises(ValueError, match="Error fetching content from URL"):
            loader.load(SourceContent("https://example.com/notfound.csv"))
