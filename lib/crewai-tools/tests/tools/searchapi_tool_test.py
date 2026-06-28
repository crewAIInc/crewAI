import os
from unittest.mock import MagicMock, patch

from crewai_tools.tools.searchapi_tool.searchapi_google_search_tool import (
    SearchApiGoogleSearchTool,
)
from crewai_tools.tools.searchapi_tool.searchapi_google_shopping_tool import (
    SearchApiGoogleShoppingTool,
)
import pytest


@pytest.fixture(autouse=True)
def mock_searchapi_api_key():
    with patch.dict(os.environ, {"SEARCHAPI_API_KEY": "test_key"}):
        yield


def test_google_search_tool_initialization():
    tool = SearchApiGoogleSearchTool()
    assert tool.name == "SearchApi Google Search"
    assert tool._api_key == "test_key"


def test_api_key_not_serialized():
    """The API key must never leak via the model's serialization."""
    tool = SearchApiGoogleSearchTool()
    assert "test_key" not in str(tool.model_dump(mode="json"))
    assert "api_key" not in tool.model_dump(mode="json")


def test_google_shopping_tool_initialization():
    tool = SearchApiGoogleShoppingTool()
    assert tool.name == "SearchApi Google Shopping"


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            SearchApiGoogleSearchTool()


@patch("crewai_tools.tools.searchapi_tool.searchapi_base_tool.requests.get")
def test_google_search_run(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "search_metadata": {"id": "abc"},
        "organic_results": [
            {"title": "T1", "link": "http://t1.com", "snippet": "S1"},
        ],
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    tool = SearchApiGoogleSearchTool()
    result = tool._run(search_query="best electric cars 2026", location="New York")

    # noisy metadata is omitted, real results are kept
    assert "organic_results" in result
    assert "search_metadata" not in result

    # request is shaped correctly
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["engine"] == "google"
    assert kwargs["params"]["q"] == "best electric cars 2026"
    assert kwargs["params"]["location"] == "New York"
    assert kwargs["headers"]["Authorization"] == "Bearer test_key"
