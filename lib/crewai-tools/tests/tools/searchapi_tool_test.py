"""Tests for SearchApiSearchTool."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools.tools.searchapi_tool.searchapi_search_tool import (
    SearchApiSearchTool,
)


@pytest.fixture(autouse=True)
def mock_searchapi_api_key():
    with patch.dict(os.environ, {"SEARCHAPI_API_KEY": "test_key"}):
        yield


class TestInitialization:
    """Test tool initialization and configuration."""

    def test_default_initialization(self):
        tool = SearchApiSearchTool()
        assert tool.name == "SearchApi Search"
        assert tool.engine == "google"
        assert tool.n_results == 10
        assert tool.country is None
        assert tool.language is None

    def test_custom_engine(self):
        tool = SearchApiSearchTool(engine="google_news")
        assert tool.engine == "google_news"

    def test_custom_parameters(self):
        tool = SearchApiSearchTool(
            engine="youtube",
            n_results=5,
            country="us",
            language="en",
        )
        assert tool.engine == "youtube"
        assert tool.n_results == 5
        assert tool.country == "us"
        assert tool.language == "en"

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Invalid engine"):
            SearchApiSearchTool(engine="invalid_engine")

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing SEARCHAPI_API_KEY"):
                SearchApiSearchTool()

    def test_api_key_not_serialized(self):
        """The API key must never leak via model serialization."""
        tool = SearchApiSearchTool()
        dumped = tool.model_dump(mode="json")
        assert "test_key" not in str(dumped)
        assert "_api_key" not in dumped

    def test_all_supported_engines(self):
        engines = [
            "google",
            "google_news",
            "google_shopping",
            "google_jobs",
            "youtube",
            "bing",
            "baidu",
        ]
        for engine in engines:
            tool = SearchApiSearchTool(engine=engine)
            assert tool.engine == engine


class TestSearchExecution:
    """Test the _run method with mocked HTTP requests."""

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_google_search(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "search_metadata": {"id": "abc"},
            "search_parameters": {"q": "test"},
            "pagination": {"next": "..."},
            "organic_results": [
                {"title": "Result 1", "link": "http://r1.com", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "http://r2.com", "snippet": "Snippet 2"},
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool()
        result = tool._run(search_query="best electric cars")

        assert "organic_results" in result
        assert len(result["organic_results"]) == 2
        assert "search_metadata" not in result
        assert "search_parameters" not in result
        assert "pagination" not in result

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_google_news_search(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "search_metadata": {"id": "def"},
            "news_results": [
                {"title": "News 1", "link": "http://n1.com", "date": "2026-01-01"},
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool(engine="google_news")
        result = tool._run(search_query="AI news")

        assert "news_results" in result
        assert "search_metadata" not in result

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_google_shopping_search(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "search_metadata": {"id": "ghi"},
            "shopping_results": [
                {"title": "Product 1", "price": "$99", "link": "http://p1.com"},
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool(engine="google_shopping")
        result = tool._run(search_query="wireless headphones")

        assert "shopping_results" in result
        assert "search_metadata" not in result

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_youtube_search(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "search_metadata": {"id": "jkl"},
            "video_results": [
                {"title": "Video 1", "link": "http://yt.com/1"},
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool(engine="youtube")
        result = tool._run(search_query="python tutorial")

        assert "video_results" in result
        assert "search_metadata" not in result


class TestRequestConstruction:
    """Test that requests are constructed correctly."""

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_request_params(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool(engine="google", n_results=5)
        tool._run(search_query="test query")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["engine"] == "google"
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["params"]["num"] == 5
        assert kwargs["headers"]["Authorization"] == "Bearer test_key"

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_request_with_location(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool()
        tool._run(search_query="coffee shops", location="San Francisco")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["location"] == "San Francisco"

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_request_with_country_and_language(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool(country="us", language="en")
        tool._run(search_query="test")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["gl"] == "us"
        assert kwargs["params"]["hl"] == "en"

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_location_not_sent_when_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool()
        tool._run(search_query="test")

        _, kwargs = mock_get.call_args
        assert "location" not in kwargs["params"]
        assert "gl" not in kwargs["params"]
        assert "hl" not in kwargs["params"]


class TestErrorHandling:
    """Test error handling for various failure modes."""

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_timeout_returns_message(self, mock_get):
        mock_get.side_effect = requests.Timeout("connection timed out")

        tool = SearchApiSearchTool()
        result = tool._run(search_query="anything")

        assert isinstance(result, str)
        assert "error occurred" in result.lower()

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_connection_error_returns_message(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("failed to connect")

        tool = SearchApiSearchTool()
        result = tool._run(search_query="anything")

        assert isinstance(result, str)
        assert "error occurred" in result.lower()

    @patch(
        "crewai_tools.tools.searchapi_tool.searchapi_search_tool.requests.get"
    )
    def test_http_error_returns_message(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
        mock_get.return_value = mock_response

        tool = SearchApiSearchTool()
        result = tool._run(search_query="anything")

        assert isinstance(result, str)
        assert "error occurred" in result.lower()

    def test_missing_search_query_raises(self):
        tool = SearchApiSearchTool()
        with pytest.raises(ValueError, match="search_query is required"):
            tool._run()


if __name__ == "__main__":
    pytest.main([__file__])
