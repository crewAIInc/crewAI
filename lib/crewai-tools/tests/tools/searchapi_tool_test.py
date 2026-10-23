import os
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from crewai_tools.tools.searchapi_tool.searchapi_tool import SearchApiTool


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    """Build a ``requests.Response``-like mock with the attributes ``_run`` uses."""
    resp = MagicMock(spec=requests_lib.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.text = text or (str(json_data) if json_data else "")
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


@pytest.fixture(autouse=True)
def _searchapi_env():
    with patch.dict(os.environ, {"SEARCHAPI_API_KEY": "test-api-key"}):
        yield


@pytest.fixture
def tool():
    return SearchApiTool()


def test_default_attributes(tool):
    assert tool.search_url == "https://www.searchapi.io/api/v1/search"
    assert tool.engine == "google"
    assert tool.n_results == 10
    assert tool.env_vars[0].name == "SEARCHAPI_API_KEY"


def test_missing_query_raises(tool):
    with pytest.raises(ValueError, match="search_query is required"):
        tool._run()


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SEARCHAPI_API_KEY"):
            SearchApiTool()._run(search_query="test")


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_key_is_sent_as_bearer_header_not_query_param(mock_get, tool):
    """The key must stay out of the query string, which SearchApi echoes back."""
    mock_get.return_value = _mock_response(json_data={"organic_results": []})

    tool._run(search_query="crewai")

    kwargs = mock_get.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"
    assert "api_key" not in kwargs["params"]


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_explicit_api_key_takes_precedence_over_env(mock_get):
    mock_get.return_value = _mock_response(json_data={})

    SearchApiTool(api_key="explicit-key")._run(search_query="crewai")

    assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer explicit-key"


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_localization_params_are_only_sent_when_set(mock_get):
    mock_get.return_value = _mock_response(json_data={})

    SearchApiTool(country="uk", locale="en", location="London,England")._run(
        search_query="fish"
    )

    params = mock_get.call_args.kwargs["params"]
    assert params == {
        "engine": "google",
        "q": "fish",
        "gl": "uk",
        "hl": "en",
        "location": "London,England",
    }


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_engine_can_be_overridden_per_call(mock_get, tool):
    mock_get.return_value = _mock_response(json_data={})

    tool._run(search_query="crewai", engine="google_news")

    assert mock_get.call_args.kwargs["params"]["engine"] == "google_news"


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_query_alias_is_accepted(mock_get, tool):
    mock_get.return_value = _mock_response(json_data={})

    tool._run(query="crewai")

    assert mock_get.call_args.kwargs["params"]["q"] == "crewai"


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_result_lists_are_capped_at_n_results(mock_get):
    mock_get.return_value = _mock_response(
        json_data={
            "organic_results": [{"position": i} for i in range(10)],
            "related_questions": [{"question": f"q{i}"} for i in range(10)],
        }
    )

    result = SearchApiTool(n_results=3)._run(search_query="crewai")

    assert len(result["organic_results"]) == 3
    # Only lists named *_results are capped; everything else passes through.
    assert len(result["related_questions"]) == 10


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_data_uris_are_dropped_from_results(mock_get, tool):
    mock_get.return_value = _mock_response(
        json_data={
            "organic_results": [
                {
                    "title": "CrewAI",
                    "link": "https://crewai.com",
                    "favicon": "data:image/png;base64," + "A" * 20000,
                }
            ],
            "inline_images": ["data:image/png;base64,AAAA", "https://img.co/a.png"],
        }
    )

    result = tool._run(search_query="crewai")

    assert result["organic_results"][0] == {
        "title": "CrewAI",
        "link": "https://crewai.com",
    }
    assert result["inline_images"] == ["https://img.co/a.png"]


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_long_strings_are_truncated(mock_get):
    mock_get.return_value = _mock_response(
        json_data={"organic_results": [{"snippet": "x" * 500}]}
    )

    result = SearchApiTool(max_string_length=100)._run(search_query="crewai")

    assert result["organic_results"][0]["snippet"] == "x" * 100 + "..."


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_empty_result_page_is_returned_not_raised(mock_get, tool):
    """A 200 carrying `error` is SearchApi reporting no results, not a failure."""
    mock_get.return_value = _mock_response(
        json_data={
            "search_metadata": {"id": "search_1", "status": "Success"},
            "error": "Google didn't return any results.",
        }
    )

    result = tool._run(search_query="asdkjhaskdjh")

    assert result["error"] == "Google didn't return any results."


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_error_status_raises_with_api_message(mock_get, tool):
    mock_get.return_value = _mock_response(
        status_code=401, json_data={"error": "Invalid API key."}
    )

    with pytest.raises(RuntimeError, match="HTTP 401.*Invalid API key."):
        tool._run(search_query="crewai")


@patch("crewai_tools.tools.searchapi_tool.searchapi_tool.requests.get")
def test_non_json_error_body_falls_back_to_text(mock_get, tool):
    resp = _mock_response(status_code=502, text="<html>Bad gateway</html>")
    resp.json.side_effect = ValueError("not json")
    mock_get.return_value = resp

    with pytest.raises(RuntimeError, match="HTTP 502.*Bad gateway"):
        tool._run(search_query="crewai")
