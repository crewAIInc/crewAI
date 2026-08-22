import os
from unittest.mock import patch

from crewai_tools.tools.crustapi_tool.crustapi_tool import CrustApiTool
import pytest


@pytest.fixture(autouse=True)
def mock_crustapi_api_key():
    with patch.dict(os.environ, {"CRUSTAPI_API_KEY": "test_key"}):
        yield


@pytest.fixture
def crustapi_tool():
    return CrustApiTool(n_results=2)


def test_crustapi_tool_initialization():
    tool = CrustApiTool()
    assert tool.n_results == 10
    assert tool.save_file is False
    assert tool.search_type == "search"
    assert tool.country == ""
    assert tool.location == ""
    assert tool.locale == ""


def test_crustapi_tool_custom_initialization():
    tool = CrustApiTool(
        n_results=5,
        save_file=True,
        search_type="news",
        country="US",
        location="New York",
        locale="en",
    )
    assert tool.n_results == 5
    assert tool.save_file is True
    assert tool.search_type == "news"
    assert tool.country == "US"
    assert tool.location == "New York"
    assert tool.locale == "en"


def test_crustapi_tool_invalid_search_type():
    tool = CrustApiTool(search_type="not_a_real_type")
    with pytest.raises(ValueError, match="Invalid search type"):
        tool.run(search_query="test query")


@patch("requests.get")
def test_crustapi_tool_search(mock_get):
    tool = CrustApiTool(n_results=2)
    mock_response = {
        "searchParameters": {"q": "test query", "type": "web"},
        "organic": [
            {
                "title": "Test Title 1",
                "link": "http://test1.com",
                "snippet": "Test Description 1",
                "position": 1,
            },
            {
                "title": "Test Title 2",
                "link": "http://test2.com",
                "snippet": "Test Description 2",
                "position": 2,
            },
        ],
        "peopleAlsoAsk": [
            {
                "question": "Test Question",
                "snippet": "Test Answer",
                "title": "Test Source",
                "link": "http://test.com",
            }
        ],
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    result = tool.run(search_query="test query")

    assert "searchParameters" in result
    assert result["searchParameters"]["q"] == "test query"
    assert result["searchParameters"]["type"] == "web"
    assert len(result["organic"]) == 2
    assert result["organic"][0]["title"] == "Test Title 1"
    assert len(result["peopleAlsoAsk"]) == 1


@patch("requests.get")
def test_crustapi_tool_news_search(mock_get):
    tool = CrustApiTool(n_results=2, search_type="news")
    mock_response = {
        "searchParameters": {"q": "test news", "type": "news"},
        "news": [
            {
                "title": "News Title 1",
                "link": "http://news1.com",
                "snippet": "News Description 1",
                "date": "2024-01-01",
                "source": "Test Source",
                "imageUrl": "http://news1.com/image.jpg",
            }
        ],
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    result = tool.run(search_query="test news")

    assert result["searchParameters"]["type"] == "news"
    assert len(result["news"]) == 1
    assert result["news"][0]["title"] == "News Title 1"


@patch("requests.get")
def test_crustapi_tool_maps_search(mock_get):
    tool = CrustApiTool(n_results=2, search_type="maps")
    mock_response = {
        "searchParameters": {"q": "coffee austin", "type": "maps"},
        "places": [
            {"title": "Cafe One", "rating": 4.8},
            {"title": "Cafe Two", "rating": 4.6},
        ],
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    result = tool.run(search_query="coffee austin")

    assert result["searchParameters"]["type"] == "maps"
    assert len(result["places"]) == 2
    assert result["places"][0]["title"] == "Cafe One"
