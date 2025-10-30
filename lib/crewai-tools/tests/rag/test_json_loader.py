import json
import os
import tempfile
from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.json_loader import JSONLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


class TestJSONLoader:
    def _create_temp_json_file(self, data) -> str:
        """Helper to write JSON data to a temporary file and return its path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            return f.name

    def _create_temp_raw_file(self, content: str) -> str:
        """Helper to write raw content to a temporary file and return its path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(content)
            return f.name

    def _load_from_path(self, path) -> LoaderResult:
        loader = JSONLoader()
        return loader.load(SourceContent(path))

    def test_load_json_dict(self):
        path = self._create_temp_json_file(
            {"name": "John", "age": 30, "items": ["a", "b", "c"]}
        )
        try:
            result = self._load_from_path(path)
            assert isinstance(result, LoaderResult)
            assert all(k in result.content for k in ["name", "John", "age", "30"])
            assert result.metadata == {"format": "json", "type": "dict", "size": 3}
            assert result.source == path
        finally:
            os.unlink(path)

    def test_load_json_list(self):
        path = self._create_temp_json_file(
            [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
            ]
        )
        try:
            result = self._load_from_path(path)
            assert result.metadata["type"] == "list"
            assert result.metadata["size"] == 2
            assert all(item in result.content for item in ["Item 1", "Item 2"])
        finally:
            os.unlink(path)

    @pytest.mark.parametrize(
        "value, expected_type",
        [
            ("simple string value", "str"),
            (42, "int"),
        ],
    )
    def test_load_json_primitives(self, value, expected_type):
        path = self._create_temp_json_file(value)
        try:
            result = self._load_from_path(path)
            assert result.metadata["type"] == expected_type
            assert result.metadata["size"] == 1
            assert str(value) in result.content
        finally:
            os.unlink(path)

    def test_load_malformed_json(self):
        path = self._create_temp_raw_file('{"invalid": json,}')
        try:
            result = self._load_from_path(path)
            assert result.metadata["format"] == "json"
            assert "parse_error" in result.metadata
            assert result.content == '{"invalid": json,}'
        finally:
            os.unlink(path)

    def test_load_empty_file(self):
        path = self._create_temp_raw_file("")
        try:
            result = self._load_from_path(path)
            assert "parse_error" in result.metadata
            assert result.content == ""
        finally:
            os.unlink(path)

    def test_load_text_input(self):
        json_text = '{"message": "hello", "count": 5}'
        loader = JSONLoader()
        result = loader.load(SourceContent(json_text))
        assert all(
            part in result.content for part in ["message", "hello", "count", "5"]
        )
        assert result.metadata["type"] == "dict"
        assert result.metadata["size"] == 2

    def test_load_complex_nested_json(self):
        data = {
            "users": [
                {"id": 1, "profile": {"name": "Alice", "settings": {"theme": "dark"}}},
                {"id": 2, "profile": {"name": "Bob", "settings": {"theme": "light"}}},
            ],
            "meta": {"total": 2, "version": "1.0"},
        }
        path = self._create_temp_json_file(data)
        try:
            result = self._load_from_path(path)
            for value in ["Alice", "Bob", "dark", "light"]:
                assert value in result.content
            assert result.metadata["size"] == 2  # top-level keys
        finally:
            os.unlink(path)

    def test_consistent_doc_id(self):
        path = self._create_temp_json_file({"test": "data"})
        try:
            result1 = self._load_from_path(path)
            result2 = self._load_from_path(path)
            assert result1.doc_id == result2.doc_id
        finally:
            os.unlink(path)

    # ------------------------------
    # URL-based tests
    # ------------------------------

    @patch("requests.get")
    def test_url_response_valid_json(self, mock_get):
        mock_get.return_value = Mock(
            text='{"key": "value", "number": 123}',
            json=Mock(return_value={"key": "value", "number": 123}),
            raise_for_status=Mock(),
        )

        loader = JSONLoader()
        result = loader.load(SourceContent("https://api.example.com/data.json"))

        assert all(val in result.content for val in ["key", "value", "number", "123"])
        headers = mock_get.call_args[1]["headers"]
        assert "application/json" in headers["Accept"]
        assert "crewai-tools JSONLoader" in headers["User-Agent"]

    @patch("requests.get")
    def test_url_response_not_json(self, mock_get):
        mock_get.return_value = Mock(
            text='{"key": "value"}',
            json=Mock(side_effect=ValueError("Not JSON")),
            raise_for_status=Mock(),
        )

        loader = JSONLoader()
        result = loader.load(SourceContent("https://example.com/data.json"))
        assert all(part in result.content for part in ["key", "value"])

    @patch("requests.get")
    def test_url_with_custom_headers(self, mock_get):
        mock_get.return_value = Mock(
            text='{"data": "test"}',
            json=Mock(return_value={"data": "test"}),
            raise_for_status=Mock(),
        )
        headers = {"Authorization": "Bearer token", "Custom-Header": "value"}

        loader = JSONLoader()
        loader.load(SourceContent("https://api.example.com/data.json"), headers=headers)

        assert mock_get.call_args[1]["headers"] == headers

    @patch("requests.get")
    def test_url_network_failure(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        loader = JSONLoader()
        with pytest.raises(ValueError, match="Error fetching content from URL"):
            loader.load(SourceContent("https://api.example.com/data.json"))

    @patch("requests.get")
    def test_url_http_error(self, mock_get):
        mock_get.return_value = Mock(
            raise_for_status=Mock(side_effect=Exception("404"))
        )
        loader = JSONLoader()
        with pytest.raises(ValueError, match="Error fetching content from URL"):
            loader.load(SourceContent("https://api.example.com/404.json"))
