import os
from unittest.mock import patch

from crewai_tools.tools.serpex_tool.serpex_tool import SerpexTool
import pytest


@pytest.fixture(autouse=True)
def mock_serpex_api_key():
    with patch.dict(os.environ, {"SERPEX_API_KEY": "test_key"}):
        yield


@pytest.fixture
def serpex_tool():
    return SerpexTool(n_results=2)


def test_serpex_tool_initialization():
    tool = SerpexTool()
    assert tool.n_results == 10
    assert tool.save_file is False
    assert tool.engine == "auto"
    assert tool.time_range == "all"


def test_serpex_tool_custom_initialization():
    tool = SerpexTool(
        n_results=5,
        save_file=True,
        engine="google",
        time_range="week",
    )
    assert tool.n_results == 5
    assert tool.save_file is True
    assert tool.engine == "google"
    assert tool.time_range == "week"


@patch("requests.get")
def test_serpex_tool_search(mock_get):
    tool = SerpexTool(n_results=2)
    mock_response = {
        "metadata": {
            "number_of_results": 17,
            "response_time": 0.5,
            "timestamp": "2025-10-26T12:00:00Z",
            "credits_used": 1,
        },
        "id": "test_id",
        "query": "test query",
        "engines": ["google"],
        "results": [
            {
                "title": "Test Title 1",
                "url": "http://test1.com",
                "snippet": "Test Description 1",
                "position": 1,
                "engine": "google",
            },
            {
                "title": "Test Title 2",
                "url": "http://test2.com",
                "snippet": "Test Description 2",
                "position": 2,
                "engine": "google",
            },
        ],
        "answers": [],
        "suggestions": ["related query 1", "related query 2"],
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    result = tool.run(search_query="test query")

    assert "query" in result
    assert result["query"] == "test query"
    assert "engines" in result
    assert result["engines"] == ["google"]
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Test Title 1"
    assert result["metadata"]["number_of_results"] == 17
    assert result["metadata"]["credits_used"] == 1


@patch("requests.get")
def test_serpex_tool_multi_engine_search(mock_get):
    tool = SerpexTool(n_results=2, engine="auto")
    mock_response = {
        "metadata": {
            "number_of_results": 10,
            "response_time": 0.3,
            "timestamp": "2025-10-26T12:00:00Z",
            "credits_used": 1,
        },
        "query": "test query",
        "engines": ["google", "bing"],
        "results": [
            {
                "title": "Result 1",
                "url": "http://result1.com",
                "snippet": "Description 1",
                "position": 1,
                "engine": "google",
            }
        ],
        "suggestions": [],
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    result = tool.run(search_query="test query", engine="auto")

    assert result["engines"] == ["google", "bing"]
    assert result["metadata"]["number_of_results"] == 10


@patch("requests.get")
def test_serpex_tool_time_filtered_search(mock_get):
    tool = SerpexTool(n_results=2, time_range="week")
    mock_response = {
        "metadata": {
            "number_of_results": 5,
            "response_time": 0.4,
            "timestamp": "2025-10-26T12:00:00Z",
            "credits_used": 1,
        },
        "query": "recent news",
        "engines": ["google"],
        "results": [
            {
                "title": "Recent News",
                "url": "http://news.com",
                "snippet": "Latest updates",
                "position": 1,
                "engine": "google",
                "published_date": "2025-10-25",
            }
        ],
        "suggestions": [],
    }
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.status_code = 200

    result = tool.run(search_query="recent news", time_range="week")

    assert result["results"][0]["published_date"] == "2025-10-25"


def test_serpex_tool_validate_engine():
    tool = SerpexTool()
    assert tool._validate_engine("google") == "google"
    assert tool._validate_engine("AUTO") == "auto"
    assert tool._validate_engine("DuckDuckGo") == "duckduckgo"

    with pytest.raises(ValueError, match="Invalid engine"):
        tool._validate_engine("invalid_engine")


def test_serpex_tool_validate_time_range():
    tool = SerpexTool()
    assert tool._validate_time_range("day") == "day"
    assert tool._validate_time_range("WEEK") == "week"
    assert tool._validate_time_range("Month") == "month"

    with pytest.raises(ValueError, match="Invalid time_range"):
        tool._validate_time_range("invalid_range")


@patch("requests.get")
def test_serpex_tool_error_handling(mock_get):
    tool = SerpexTool()
    mock_get.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        tool.run(search_query="test query")


def test_serpex_tool_missing_search_query():
    tool = SerpexTool()

    with pytest.raises(ValueError, match="search_query is required"):
        tool.run()


@patch("requests.get")
def test_serpex_tool_all_engines(mock_get):
    """Test all supported engines"""
    engines = ["google", "bing", "duckduckgo", "brave", "yahoo", "yandex"]
    tool = SerpexTool()

    for engine in engines:
        mock_response = {
            "metadata": {"number_of_results": 5, "credits_used": 1},
            "query": "test",
            "engines": [engine],
            "results": [
                {
                    "title": f"Result from {engine}",
                    "url": "http://test.com",
                    "snippet": "Test",
                    "position": 1,
                    "engine": engine,
                }
            ],
            "suggestions": [],
        }
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        result = tool.run(search_query="test", engine=engine)
        assert result["engines"] == [engine]
        assert result["results"][0]["engine"] == engine
