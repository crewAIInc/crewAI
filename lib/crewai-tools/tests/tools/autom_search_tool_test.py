import json
import os
from unittest.mock import MagicMock, patch

from crewai_tools.tools.autom_search_tool.autom_search_tool import AutomSearchTool
import pytest
import requests


@pytest.fixture(autouse=True)
def autom_api_key():
    with patch.dict(os.environ, {"AUTOM_API_KEY": "test_key"}):
        yield


def _response(status=200, body=None):
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.ok = status < 400
    response.json.return_value = body
    response.text = json.dumps(body)
    return response


ROWS = [
    {"position": 1, "title": "CrewAI", "link": "https://www.crewai.com", "snippet": "Multi-agent framework."},
    "not a result",
    {"position": 2, "title": "Docs", "link": "https://docs.crewai.com", "snippet": "Documentation."},
]


@patch("requests.get")
def test_google_search_by_default(mock_get):
    mock_get.return_value = _response(body={"organic_results": ROWS})

    result = json.loads(AutomSearchTool().run(search_query=" crewai "))

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.autom.dev/v1/google/search"
    assert kwargs["params"] == {"query": "crewai"}
    assert kwargs["headers"]["x-api-key"] == "test_key"
    assert [r["title"] for r in result] == ["CrewAI", "Docs"]
    assert result[0] == {
        "title": "CrewAI",
        "link": "https://www.crewai.com",
        "snippet": "Multi-agent framework.",
        "position": 1,
    }


@patch("requests.get")
def test_google_options_and_result_limit(mock_get):
    mock_get.return_value = _response(body={"organic_results": ROWS})

    tool = AutomSearchTool(n_results=1, country="FR", language="fr", location="Paris, France")
    result = json.loads(tool.run(search_query="crewai"))

    assert mock_get.call_args.kwargs["params"] == {
        "query": "crewai",
        "gl": "fr",
        "hl": "fr",
        "location": "Paris, France",
    }
    assert len(result) == 1


@patch("requests.get")
def test_news_uses_google_news(mock_get):
    news = [{"title": "News", "link": "https://n.example", "snippet": "s", "date": "1 hour ago", "source": "Example"}]
    mock_get.return_value = _response(body={"organic_results": news})

    result = json.loads(AutomSearchTool(engine="bing", search_type="news", location="Paris").run(search_query="ai"))

    assert mock_get.call_args.args[0] == "https://api.autom.dev/v1/google/news"
    assert "location" not in mock_get.call_args.kwargs["params"]
    assert result[0]["date"] == "1 hour ago" and result[0]["source"] == "Example"


@pytest.mark.parametrize(
    ("engine", "country_param"),
    [("bing", "cc"), ("brave", "gl"), ("duckduckgo", "gl"), ("yahoo", "gl")],
)
@patch("requests.get")
def test_other_engines(mock_get, engine, country_param):
    mock_get.return_value = _response(body={"organic_results": ROWS})

    AutomSearchTool(engine=engine, country="US", language="en").run(search_query="crewai")

    assert mock_get.call_args.args[0] == f"https://api.autom.dev/v1/{engine}/search"
    assert mock_get.call_args.kwargs["params"] == {"query": "crewai", country_param: "us"}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": "Invalid API key test_key"}, "Autom rejected the API key (HTTP 401): Invalid API key ***"),
        (401, {"error": "Rate limit exceeded"}, "Autom rate limit exceeded (HTTP 401): Rate limit exceeded"),
        (402, {"error": {"message": "No credits"}}, "Autom account is out of credits (HTTP 402): No credits"),
        (500, {"error": "Internal Error"}, "Autom API error (HTTP 500): Internal Error"),
    ],
)
@patch("requests.get")
def test_api_errors_are_returned_without_the_key(mock_get, status, body, expected):
    mock_get.return_value = _response(status=status, body=body)

    assert AutomSearchTool().run(search_query="crewai") == expected


@patch("requests.get")
def test_network_errors(mock_get):
    mock_get.side_effect = requests.Timeout()
    assert AutomSearchTool(timeout=5).run(search_query="crewai") == "Autom did not answer within 5 seconds."

    mock_get.side_effect = requests.ConnectionError("failed with test_key")
    assert AutomSearchTool().run(search_query="crewai") == "Could not reach Autom: failed with ***"


@patch("requests.get")
def test_no_results(mock_get):
    mock_get.return_value = _response(body={"organic_results": []})
    assert AutomSearchTool().run(search_query="crewai") == "No results for: crewai"


@patch("requests.get")
def test_missing_key_and_empty_query_send_no_request(mock_get):
    assert AutomSearchTool().run(search_query="  ") == "The search query is empty."
    with patch.dict(os.environ, {"AUTOM_API_KEY": ""}):
        assert "AUTOM_API_KEY is not set" in AutomSearchTool().run(search_query="crewai")
    mock_get.assert_not_called()


def test_declares_its_env_var():
    assert [env.name for env in AutomSearchTool().env_vars] == ["AUTOM_API_KEY"]


@pytest.mark.parametrize("n_results", [0, 11])
def test_n_results_stays_within_one_page(n_results):
    with pytest.raises(ValueError):
        AutomSearchTool(n_results=n_results)
