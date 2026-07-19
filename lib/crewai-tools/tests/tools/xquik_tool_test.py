import os
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from crewai_tools.tools.xquik_tool.xquik_tool import (
    XQUIK_API_CONTRACT,
    XquikGetTrendsTool,
    XquikGetTweetTool,
    XquikGetUserTool,
    XquikGetUserTweetsTool,
    XquikSearchTweetsTool,
)


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    """Build a requests-compatible response double."""
    response = MagicMock(spec=requests_lib.Response)
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.text = text or (str(json_data) if json_data else "")
    response.json.return_value = json_data if json_data is not None else {}
    return response


@pytest.fixture(autouse=True)
def _xquik_env():
    """Provide a deterministic API key for each test."""
    with patch.dict(os.environ, {"XQUIK_API_KEY": "test-api-key"}):
        yield


def test_instantiation_with_env_var():
    """Read authentication and defaults from the environment."""
    tool = XquikSearchTweetsTool()
    assert tool.api_key == "test-api-key"
    assert tool.base_url == "https://xquik.com/api/v1"
    assert tool.timeout == 30


def test_instantiation_with_explicit_args():
    """Prefer explicit request configuration over defaults."""
    tool = XquikGetUserTool(
        api_key="explicit-key",
        base_url="https://example.test/api/v1/",
        timeout=5,
    )

    assert tool.api_key == "explicit-key"
    assert tool.base_url == "https://example.test/api/v1"
    assert tool.timeout == 5


def test_missing_api_key_raises():
    """Reject construction without authentication."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="XQUIK_API_KEY"):
            XquikSearchTweetsTool()


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_search_tweets_calls_xquik_api(mock_get):
    """Send normalized search parameters and authentication headers."""
    tool = XquikSearchTweetsTool()
    mock_get.return_value = _mock_response(
        json_data={
            "tweets": [{"id": "1", "text": "hello"}],
            "has_more": False,
        }
    )

    result = tool.run(search_query="from:crewai AI agents", limit=10, query_type="Top")

    assert result["tweets"][0]["id"] == "1"
    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["headers"]["x-api-key"] == "test-api-key"
    assert kwargs["headers"]["xquik-api-contract"] == XQUIK_API_CONTRACT
    assert kwargs["params"] == {
        "q": "from:crewai AI agents",
        "limit": 10,
        "queryType": "Top",
    }
    assert kwargs["timeout"] == 30


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_get_tweet_encodes_tweet_id(mock_get):
    """Percent-encode reserved characters in post identifiers."""
    tool = XquikGetTweetTool()
    mock_get.return_value = _mock_response(json_data={"tweet": {"id": "12 /?%"}})

    result = tool.run(tweet_id="12 /?%")

    assert result["tweet"]["id"] == "12 /?%"
    assert mock_get.call_args.args[0] == (
        "https://xquik.com/api/v1/x/tweets/12%20%2F%3F%25"
    )
    assert "12 /?%" not in mock_get.call_args.args[0]


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_get_user_strips_leading_at_and_encodes_path(mock_get):
    """Normalize handles and percent-encode reserved path characters."""
    tool = XquikGetUserTool()
    mock_get.return_value = _mock_response(json_data={"username": "crew ai/?"})

    result = tool.run(user="@crew ai/?")

    assert result["username"] == "crew ai/?"
    assert mock_get.call_args.args[0] == (
        "https://xquik.com/api/v1/x/users/crew%20ai%2F%3F"
    )
    assert "crew ai/?" not in mock_get.call_args.args[0]


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_get_user_tweets_sends_optional_flags(mock_get):
    """Forward enabled timeline options using API parameter names."""
    tool = XquikGetUserTweetsTool()
    mock_get.return_value = _mock_response(json_data={"tweets": []})

    result = tool.run(
        user="crew_ai",
        cursor="cursor-1",
        include_replies=True,
        include_parent_tweet=True,
    )

    assert result["tweets"] == []
    _, kwargs = mock_get.call_args
    assert kwargs["params"] == {
        "cursor": "cursor-1",
        "includeReplies": "true",
        "includeParentTweet": "true",
    }


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_get_trends_calls_region_endpoint(mock_get):
    """Send bounded region and result-count parameters."""
    tool = XquikGetTrendsTool()
    mock_get.return_value = _mock_response(
        json_data={"trends": [{"name": "#AI"}], "count": 1, "woeid": 1}
    )

    result = tool.run(woeid=1, count=1)

    assert result["trends"][0]["name"] == "#AI"
    assert mock_get.call_args.args[0] == "https://xquik.com/api/v1/x/trends"
    assert mock_get.call_args.kwargs["params"] == {"woeid": 1, "count": 1}


def test_validation_rejects_empty_inputs_and_invalid_ranges():
    """Reject empty identifiers and out-of-range values locally."""
    search_tool = XquikSearchTweetsTool()
    tweet_tool = XquikGetTweetTool()
    user_tool = XquikGetUserTool()
    trends_tool = XquikGetTrendsTool()

    with pytest.raises(ValueError, match="search_query is required"):
        search_tool.run(search_query=" ")

    with pytest.raises(ValueError, match="tweet_id is required"):
        tweet_tool.run(tweet_id="@")

    with pytest.raises(ValueError, match="user is required"):
        user_tool.run(user="@")

    with pytest.raises(ValueError, match="less than or equal to 200"):
        search_tool.run(search_query="AI", limit=201)

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        trends_tool.run(woeid=0)

    with pytest.raises(ValueError, match="less than or equal to 50"):
        trends_tool.run(count=51)


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_api_errors_include_status_and_response_body(mock_get):
    """Expose bounded API error context without losing status."""
    tool = XquikGetTweetTool()
    mock_get.return_value = _mock_response(
        status_code=404,
        json_data={"error": "tweet_not_found", "message": "Tweet not found."},
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        tool.run(tweet_id="missing")


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_api_errors_bound_json_response_body(mock_get):
    """Bound structured API error details before exposing them."""
    tool = XquikGetTweetTool()
    mock_get.return_value = _mock_response(
        status_code=502,
        json_data={"message": "x" * 1_000},
    )

    with pytest.raises(RuntimeError) as error:
        tool.run(tweet_id="missing")

    assert len(str(error.value).split(": ", maxsplit=1)[1]) == 500


@patch("crewai_tools.tools.xquik_tool.xquik_tool.safe_get")
def test_request_exceptions_are_wrapped(mock_get):
    """Wrap transport errors with Xquik request context."""
    tool = XquikGetTrendsTool()
    mock_get.side_effect = requests_lib.Timeout("timed out")

    with pytest.raises(RuntimeError, match="Xquik API request failed"):
        tool.run()
