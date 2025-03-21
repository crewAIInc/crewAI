from unittest.mock import patch
import pytest
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
import os


@pytest.fixture(autouse=True)
def mock_serper_api_key():
    with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
        yield


@pytest.fixture
def serper_tool():
    return SerperDevTool(n_results=2)


def test_serper_tool_initialization():
    tool = SerperDevTool()
    assert tool.n_results == 10
    assert tool.save_file is False
    assert tool.search_type == "search"
    assert tool.country == ""
    assert tool.location == ""
    assert tool.locale == ""


def test_serper_tool_custom_initialization():
    tool = SerperDevTool(
        n_results=5,
        save_file=True,
        search_type="news",
        country="US",
        location="New York",
        locale="en"
    )
    assert tool.n_results == 5
    assert tool.save_file is True
    assert tool.search_type == "news"
    assert tool.country == "US"
    assert tool.location == "New York"
    assert tool.locale == "en"


@patch("requests.post")
def test_serper_tool_search(mock_post):
    tool = SerperDevTool(n_results=2)
    mock_response = {
        "searchParameters": {
            "q": "test query",
            "type": "search"
        },
        "organic": [
            {
                "title": "Test Title 1",
                "link": "http://test1.com",
                "snippet": "Test Description 1",
                "position": 1
            },
            {
                "title": "Test Title 2",
                "link": "http://test2.com",
                "snippet": "Test Description 2",
                "position": 2
            }
        ],
        "peopleAlsoAsk": [
            {
                "question": "Test Question",
                "snippet": "Test Answer",
                "title": "Test Source",
                "link": "http://test.com"
            }
        ]
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="test query")

    assert "searchParameters" in result
    assert result["searchParameters"]["q"] == "test query"
    assert len(result["organic"]) == 2
    assert result["organic"][0]["title"] == "Test Title 1"


@patch("requests.post")
def test_serper_tool_news_search(mock_post):
    tool = SerperDevTool(n_results=2, search_type="news")
    mock_response = {
        "searchParameters": {
            "q": "test news",
            "type": "news"
        },
        "news": [
            {
                "title": "News Title 1",
                "link": "http://news1.com",
                "snippet": "News Description 1",
                "date": "2024-01-01",
                "source": "News Source 1",
                "imageUrl": "http://image1.com"
            }
        ]
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = tool.run(search_query="test news")

    assert "news" in result
    assert len(result["news"]) == 1
    assert result["news"][0]["title"] == "News Title 1"


@patch("requests.post")
def test_serper_tool_with_location_params(mock_post):
    tool = SerperDevTool(
        n_results=2,
        country="US",
        location="New York",
        locale="en"
    )

    tool.run(search_query="test")

    called_payload = mock_post.call_args.kwargs["json"]
    assert called_payload["gl"] == "US"
    assert called_payload["location"] == "New York"
    assert called_payload["hl"] == "en"


def test_invalid_search_type():
    tool = SerperDevTool()
    with pytest.raises(ValueError) as exc_info:
        tool.run(search_query="test", search_type="invalid")
    assert "Invalid search type" in str(exc_info.value)


@patch("requests.post")
def test_api_error_handling(mock_post):
    tool = SerperDevTool()
    mock_post.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc_info:
        tool.run(search_query="test")
    assert "API Error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
