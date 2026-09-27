import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools.tools.searchapi_tool import (
    SearchApiGoogleSearchTool,
    SearchApiGoogleShoppingTool,
)


@pytest.fixture(autouse=True)
def searchapi_env():
    with patch.dict(os.environ, {"SEARCHAPI_API_KEY": "test_searchapi_key"}):
        yield


def test_missing_api_key_raises_value_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Missing API key"):
            SearchApiGoogleSearchTool()

        with pytest.raises(ValueError, match="Missing API key"):
            SearchApiGoogleShoppingTool()


def test_non_https_url_raises_value_error():
    with pytest.raises(ValueError, match="URL scheme must be HTTPS"):
        SearchApiGoogleSearchTool(url="http://www.searchapi.io/api/v1/search")

    tool = SearchApiGoogleSearchTool()
    tool.url = "http://www.searchapi.io/api/v1/search"
    with pytest.raises(ValueError, match="URL scheme must be HTTPS"):
        tool.run(search_query="test")


def test_invalid_host_url_raises_value_error():
    with pytest.raises(ValueError, match="URL host must be searchapi.io or www.searchapi.io"):
        SearchApiGoogleSearchTool(url="https://attacker.com/api/v1/search")

    tool = SearchApiGoogleSearchTool()
    tool.url = "https://evil.com/search"
    with pytest.raises(ValueError, match="URL host must be searchapi.io or www.searchapi.io"):
        tool.run(search_query="test")


@patch("requests.get")
def test_searchapi_google_search_tool_run(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "search_metadata": {"id": "123"},
        "search_parameters": {"q": "crewAI framework"},
        "organic_results": [{"title": "crewAI", "link": "https://crewai.com"}],
    }
    mock_get.return_value = mock_response

    tool = SearchApiGoogleSearchTool()
    result = tool.run(search_query="crewAI framework", location="Austin, TX")

    mock_get.assert_called_once_with(
        "https://www.searchapi.io/api/v1/search",
        headers={"Authorization": "Bearer test_searchapi_key"},
        params={"engine": "google", "q": "crewAI framework", "location": "Austin, TX"},
        timeout=30,
    )
    assert result == {
        "organic_results": [{"title": "crewAI", "link": "https://crewai.com"}]
    }


@patch("requests.get")
def test_searchapi_google_shopping_tool_run(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "search_metadata": {"id": "456"},
        "shopping_results": [{"title": "Laptop", "price": "$999"}],
    }
    mock_get.return_value = mock_response

    tool = SearchApiGoogleShoppingTool()
    result = tool.run(search_query="laptop")

    mock_get.assert_called_once_with(
        "https://www.searchapi.io/api/v1/search",
        headers={"Authorization": "Bearer test_searchapi_key"},
        params={"engine": "google_shopping", "q": "laptop"},
        timeout=30,
    )
    assert result == {
        "shopping_results": [{"title": "Laptop", "price": "$999"}]
    }


@patch("requests.get")
def test_searchapi_request_exception(mock_get):
    mock_get.side_effect = requests.RequestException("Connection error")

    tool = SearchApiGoogleSearchTool()
    result = tool.run(search_query="crewAI")

    assert "An error occurred: Connection error" in result
