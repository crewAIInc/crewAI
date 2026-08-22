import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from crewai_tools.tools.searchapi_search_tool.searchapi_search_tool import (
    SearchApiSearchTool,
)


def _mock_response(json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests_lib.Response)
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


@pytest.fixture(autouse=True)
def _searchapi_env():
    with patch.dict(os.environ, {"SEARCHAPI_API_KEY": "test-api-key"}):
        yield


def test_instantiation_with_env_var():
    tool = SearchApiSearchTool()
    assert tool.api_key == "test-api-key"


def test_instantiation_with_explicit_key():
    tool = SearchApiSearchTool(api_key="explicit-key")
    assert tool.api_key == "explicit-key"


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SEARCHAPI_API_KEY"):
            SearchApiSearchTool()


def test_default_attributes():
    tool = SearchApiSearchTool()
    assert tool.engine == "google"
    assert tool.n_results == 10
    assert tool.search_url == "https://www.searchapi.io/api/v1/search"


def test_custom_engine_and_n_results():
    tool = SearchApiSearchTool(engine="bing", n_results=5)
    assert tool.engine == "bing"
    assert tool.n_results == 5


@patch(
    "crewai_tools.tools.searchapi_search_tool.searchapi_search_tool.requests.get"
)
def test_run_builds_request_and_parses_results(mock_get):
    data = {
        "organic_results": [
            {
                "position": 1,
                "title": "CrewAI",
                "link": "https://crewai.com",
                "snippet": "AI agent framework",
            }
        ]
    }
    mock_get.return_value = _mock_response(json_data=data)

    tool = SearchApiSearchTool(engine="bing", n_results=3)
    result = tool._run(query="crewai")

    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["params"] == {"engine": "bing", "q": "crewai", "num": 3}
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    parsed = json.loads(result)
    assert parsed == [
        {
            "title": "CrewAI",
            "link": "https://crewai.com",
            "snippet": "AI agent framework",
            "position": 1,
        }
    ]


@patch(
    "crewai_tools.tools.searchapi_search_tool.searchapi_search_tool.requests.get"
)
def test_run_skips_results_missing_title_or_link(mock_get):
    data = {
        "organic_results": [
            {"title": "No link", "snippet": "x"},
            {"link": "https://no-title.com", "snippet": "y"},
            {"title": "Good", "link": "https://good.com"},
        ]
    }
    mock_get.return_value = _mock_response(json_data=data)

    parsed = json.loads(SearchApiSearchTool()._run(query="test"))
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Good"


@patch(
    "crewai_tools.tools.searchapi_search_tool.searchapi_search_tool.requests.get"
)
def test_run_handles_empty_results(mock_get):
    mock_get.return_value = _mock_response(json_data={})
    assert json.loads(SearchApiSearchTool()._run(query="test")) == []


@patch(
    "crewai_tools.tools.searchapi_search_tool.searchapi_search_tool.requests.get"
)
def test_run_returns_error_string_on_request_exception(mock_get):
    mock_get.side_effect = requests_lib.exceptions.ConnectionError("refused")
    result = SearchApiSearchTool()._run(query="test")
    assert result.startswith("Error performing search:")


@patch(
    "crewai_tools.tools.searchapi_search_tool.searchapi_search_tool.requests.get"
)
def test_run_surfaces_api_error_body_on_http_error(mock_get):
    resp = _mock_response(json_data={"error": "Invalid API key"})
    http_error = requests_lib.exceptions.HTTPError("401 Client Error")
    http_error.response = resp
    resp.raise_for_status.side_effect = http_error
    mock_get.return_value = resp

    result = SearchApiSearchTool()._run(query="test")
    assert result.startswith("Error performing search:")
    assert "Invalid API key" in result


@patch(
    "crewai_tools.tools.searchapi_search_tool.searchapi_search_tool.requests.get"
)
def test_run_handles_http_error_with_non_json_body(mock_get):
    resp = _mock_response()
    resp.json.side_effect = ValueError("no json")
    http_error = requests_lib.exceptions.HTTPError("500 Server Error")
    http_error.response = resp
    resp.raise_for_status.side_effect = http_error
    mock_get.return_value = resp

    result = SearchApiSearchTool()._run(query="test")
    assert result.startswith("Error performing search:")
    assert "500 Server Error" in result
