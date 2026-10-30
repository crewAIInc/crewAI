import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools import AnySearchTool


SUCCESS_BODY = {
    "code": 0,
    "message": "success",
    "request_id": "req-123",
    "data": {
        "results": [
            {
                "title": "CrewAI",
                "url": "https://github.com/crewAIInc/crewAI",
                "snippet": "Framework for orchestrating role-playing agents.",
                "content": "CrewAI is a lean multi-agent framework.",
            }
        ],
        "metadata": {"total_results": 1, "search_time_ms": 42},
    },
}


def _mock_response(body, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    response.raise_for_status.return_value = None
    return response


@pytest.fixture(autouse=True)
def _clean_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANYSEARCH_API_KEY", None)
        yield


def test_default_configuration():
    """Verifies the default attribute values of a fresh tool instance."""
    tool = AnySearchTool()
    assert tool.name == "AnySearch Web Search"
    assert tool.search_url == "https://api.anysearch.com/v1/search"
    assert tool.max_results == 10
    assert tool.result_format == "json"
    assert tool.api_key is None


def test_env_vars_declare_optional_api_key():
    """Verifies ANYSEARCH_API_KEY is declared as an optional environment variable."""
    tool = AnySearchTool()
    env_var = next(v for v in tool.env_vars if v.name == "ANYSEARCH_API_KEY")
    assert env_var.required is False


@patch("requests.post")
def test_anonymous_request_omits_authorization_header(mock_post):
    """Verifies no Authorization header is sent when no API key is configured."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    AnySearchTool().run(query="crewai")

    headers = mock_post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


@pytest.mark.parametrize("blank_key", ["", "   "])
def test_blank_api_key_is_normalized_to_none(blank_key):
    """Verifies a blank api_key argument is normalized to None."""
    assert AnySearchTool(api_key=blank_key).api_key is None


@pytest.mark.parametrize("blank_key", ["", "   "])
@patch("requests.post")
def test_blank_api_key_omits_authorization_header(mock_post, blank_key):
    """Verifies a blank api_key never produces an empty Bearer credential."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    AnySearchTool(api_key=blank_key).run(query="crewai")

    assert "Authorization" not in mock_post.call_args.kwargs["headers"]


@pytest.mark.parametrize("blank_key", ["", "   "])
@patch("requests.post")
def test_blank_env_api_key_omits_authorization_header(mock_post, blank_key):
    """Verifies a blank ANYSEARCH_API_KEY value is treated as unset."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    with patch.dict(os.environ, {"ANYSEARCH_API_KEY": blank_key}):
        AnySearchTool().run(query="crewai")

    assert "Authorization" not in mock_post.call_args.kwargs["headers"]


@patch("requests.post")
def test_api_key_from_environment_is_sent(mock_post):
    """Verifies the environment API key is sent as a Bearer token."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    with patch.dict(os.environ, {"ANYSEARCH_API_KEY": "test-key"}):
        AnySearchTool().run(query="crewai")

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key"


@patch("requests.post")
def test_api_key_argument_is_trimmed_and_sent(mock_post):
    """Verifies a surrounding-whitespace api_key argument is trimmed before use."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    AnySearchTool(api_key="  test-key  ").run(query="crewai")

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key"


def test_api_key_is_masked_in_repr_and_dump():
    """Verifies the API key is masked in repr() and model_dump() output."""
    with patch.dict(os.environ, {"ANYSEARCH_API_KEY": "test-key"}):
        tool = AnySearchTool()
    assert "test-key" not in repr(tool)
    assert "test-key" not in str(tool.model_dump())


@patch("requests.post")
def test_payload_and_result_mapping(mock_post):
    """Verifies the request payload and the JSON output mapping of results."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    output = AnySearchTool(max_results=3).run(query="crewai")

    payload = mock_post.call_args.kwargs["json"]
    assert payload == {"query": "crewai", "max_results": 3, "format": "json"}

    parsed = json.loads(output)
    assert parsed["query"] == "crewai"
    assert parsed["results"][0]["url"] == "https://github.com/crewAIInc/crewAI"
    assert set(parsed["results"][0]) == {"title", "url", "snippet", "content"}


@patch("requests.post")
def test_max_results_is_clamped_to_api_limit(mock_post):
    """Verifies max_results is clamped to the API limit at runtime."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    tool = AnySearchTool()
    tool.__dict__["max_results"] = 999  # bypass validation to assert runtime clamp
    tool.run(query="crewai")

    assert mock_post.call_args.kwargs["json"]["max_results"] == 10


def test_max_results_out_of_range_is_rejected_by_schema():
    """Verifies out-of-range max_results is rejected by schema validation."""
    with pytest.raises(ValueError):
        AnySearchTool(max_results=11)


@patch("requests.post")
def test_content_is_truncated(mock_post):
    """Verifies long result content is truncated with an ellipsis."""
    body = json.loads(json.dumps(SUCCESS_BODY))
    body["data"]["results"][0]["content"] = "x" * 5000
    mock_post.return_value = _mock_response(body)

    limit = 100
    output = AnySearchTool(max_content_length_per_result=limit).run(query="crewai")

    content = json.loads(output)["results"][0]["content"]
    assert content.endswith("...")
    # The final length must never exceed the configured limit.
    assert len(content) == limit


@pytest.mark.parametrize("limit", [1, 2])
@patch("requests.post")
def test_content_truncation_with_tiny_limit_omits_ellipsis(mock_post, limit):
    """Verifies limits smaller than the ellipsis truncate without appending '...'."""
    body = json.loads(json.dumps(SUCCESS_BODY))
    body["data"]["results"][0]["content"] = "x" * 100
    mock_post.return_value = _mock_response(body)

    output = AnySearchTool(max_content_length_per_result=limit).run(query="crewai")

    content = json.loads(output)["results"][0]["content"]
    assert len(content) == limit
    assert not content.endswith("...")


@patch("requests.post")
def test_content_truncation_at_ellipsis_boundary(mock_post):
    """Verifies a limit of exactly 3 yields '...' with length 3."""
    body = json.loads(json.dumps(SUCCESS_BODY))
    body["data"]["results"][0]["content"] = "x" * 100
    mock_post.return_value = _mock_response(body)

    output = AnySearchTool(max_content_length_per_result=3).run(query="crewai")

    content = json.loads(output)["results"][0]["content"]
    assert content == "..."
    assert len(content) == 3


@patch("requests.post")
def test_missing_optional_result_fields_default_to_empty_strings(mock_post):
    """Verifies optional result fields fall back to empty strings."""
    mock_post.return_value = _mock_response(
        {
            "code": 0,
            "message": "success",
            "data": {"results": [{"url": "https://example.com"}]},
        }
    )

    result = json.loads(AnySearchTool().run(query="crewai"))["results"][0]
    assert result == {
        "title": "",
        "url": "https://example.com",
        "snippet": "",
        "content": "",
    }


@patch("requests.post")
def test_business_error_code_raises(mock_post):
    """Verifies a non-zero API code raises RuntimeError with the message."""
    mock_post.return_value = _mock_response(
        {"code": 402, "message": "quota_exhausted", "data": None}
    )

    with pytest.raises(RuntimeError, match="quota_exhausted") as exc_info:
        AnySearchTool().run(query="crewai")

    err = str(exc_info.value)
    assert "code=402" in err
    # No request_id in this fixture: fall back to the "unknown" placeholder.
    assert "request_id=unknown" in err


@patch("requests.post")
def test_auth_error_code_raises(mock_post):
    """Verifies an authentication failure reported in the body raises RuntimeError."""
    mock_post.return_value = _mock_response(
        {"code": -1, "message": "Invalid API key.", "request_id": "req-401"}
    )

    with pytest.raises(RuntimeError, match="Invalid API key") as exc_info:
        AnySearchTool(api_key="wrong-key").run(query="crewai")

    err = str(exc_info.value)
    assert "code=-1" in err
    assert "request_id=req-401" in err


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
@patch("requests.post")
def test_http_error_propagates(mock_post, status_code):
    """Verifies HTTP errors propagate from the request call."""
    response = _mock_response({}, status_code=status_code)
    response.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Error")
    mock_post.return_value = response

    with pytest.raises(requests.HTTPError):
        AnySearchTool().run(query="crewai")


@patch("requests.post")
def test_timeout_propagates(mock_post):
    """Verifies a request timeout propagates to the caller."""
    mock_post.side_effect = requests.Timeout("timed out")

    with pytest.raises(requests.Timeout):
        AnySearchTool(timeout=1).run(query="crewai")


@patch("requests.post")
def test_connection_error_propagates(mock_post):
    """Verifies a connection failure propagates to the caller."""
    mock_post.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(requests.ConnectionError):
        AnySearchTool().run(query="crewai")


@patch("requests.post")
def test_timeout_value_is_passed_to_request(mock_post):
    """Verifies the configured timeout is forwarded to the HTTP call."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    AnySearchTool(timeout=7).run(query="crewai")

    assert mock_post.call_args.kwargs["timeout"] == 7


@patch("requests.post")
def test_empty_results_returns_empty_list(mock_post):
    """Verifies an empty result set is returned as an empty JSON list."""
    mock_post.return_value = _mock_response(
        {"code": 0, "message": "success", "data": {"results": []}}
    )

    assert json.loads(AnySearchTool().run(query="crewai"))["results"] == []


@patch("requests.post")
def test_non_json_body_raises(mock_post):
    """Verifies a non-JSON response body raises RuntimeError."""
    response = _mock_response(None)
    response.json.side_effect = ValueError("Expecting value")
    mock_post.return_value = response

    with pytest.raises(RuntimeError, match="not valid JSON"):
        AnySearchTool().run(query="crewai")


@pytest.mark.parametrize("body", [[], [{"code": 0}], "ok", 1])
@patch("requests.post")
def test_non_object_body_raises(mock_post, body):
    """Verifies a top-level non-object body raises RuntimeError."""
    mock_post.return_value = _mock_response(body)

    with pytest.raises(RuntimeError, match="must be an object"):
        AnySearchTool().run(query="crewai")


@pytest.mark.parametrize("code", [False, True, None, "0", 0.0])
@patch("requests.post")
def test_non_integer_code_raises(mock_post, code):
    """Verifies a non-integer 'code' is rejected instead of read as success."""
    mock_post.return_value = _mock_response(
        {"code": code, "message": "success", "data": {"results": []}}
    )

    with pytest.raises(RuntimeError, match="'code' must be an integer"):
        AnySearchTool().run(query="crewai")


@patch("requests.post")
def test_malformed_data_missing_raises(mock_post):
    """Verifies a missing 'data' field raises RuntimeError instead of empty results."""
    mock_post.return_value = _mock_response({"code": 0, "message": "success"})

    with pytest.raises(RuntimeError, match="malformed response"):
        AnySearchTool().run(query="crewai")


@patch("requests.post")
def test_malformed_data_wrong_type_raises(mock_post):
    """Verifies a non-dict 'data' field raises RuntimeError."""
    mock_post.return_value = _mock_response(
        {"code": 0, "message": "success", "data": None}
    )

    with pytest.raises(RuntimeError, match="malformed response"):
        AnySearchTool().run(query="crewai")


@patch("requests.post")
def test_malformed_results_missing_raises(mock_post):
    """Verifies a missing 'data.results' field raises RuntimeError."""
    mock_post.return_value = _mock_response(
        {"code": 0, "message": "success", "data": {"metadata": {}}}
    )

    with pytest.raises(RuntimeError, match="malformed response"):
        AnySearchTool().run(query="crewai")


@patch("requests.post")
def test_malformed_results_wrong_type_raises(mock_post):
    """Verifies a non-list 'data.results' field raises RuntimeError."""
    mock_post.return_value = _mock_response(
        {"code": 0, "message": "success", "data": {"results": {"items": []}}}
    )

    with pytest.raises(RuntimeError, match="malformed response"):
        AnySearchTool().run(query="crewai")


@pytest.mark.parametrize("item", [None, "https://example.com", 1, []])
@patch("requests.post")
def test_non_object_result_item_raises(mock_post, item):
    """Verifies a non-object result entry raises instead of being dropped."""
    mock_post.return_value = _mock_response(
        {"code": 0, "message": "success", "data": {"results": [item]}}
    )

    with pytest.raises(RuntimeError, match="must be an object"):
        AnySearchTool().run(query="crewai")


@pytest.mark.parametrize("item", [{}, {"title": "No URL"}, {"url": ""}, {"url": None}])
@patch("requests.post")
def test_result_item_without_url_raises(mock_post, item):
    """Verifies a result entry without a usable URL raises RuntimeError."""
    mock_post.return_value = _mock_response(
        {"code": 0, "message": "success", "data": {"results": [item]}}
    )

    with pytest.raises(RuntimeError, match="missing a valid 'url'"):
        AnySearchTool().run(query="crewai")

@pytest.mark.parametrize("timeout", [0, -1, -30])
def test_non_positive_timeout_is_rejected_by_schema(timeout):
    """Verifies a non-positive timeout is rejected by schema validation."""
    with pytest.raises(ValueError):
        AnySearchTool(timeout=timeout)


@pytest.mark.parametrize("limit", [0, -1, -1000])
def test_non_positive_content_limit_is_rejected_by_schema(limit):
    """Verifies a non-positive content limit is rejected by schema validation."""
    with pytest.raises(ValueError):
        AnySearchTool(max_content_length_per_result=limit)

@patch("requests.post")
def test_api_key_requires_https_endpoint(mock_post):
    """Verifies an API key is never sent to a non-HTTPS endpoint."""
    tool = AnySearchTool(
        api_key="secret-key",
        search_url="http://insecure.example.com/v1/search",
    )

    with pytest.raises(ValueError, match="non-HTTPS"):
        tool.run(query="crewai")

    # The request must not be issued at all: otherwise the key leaks.
    mock_post.assert_not_called()


@patch("requests.post")
def test_anonymous_request_allows_http_endpoint(mock_post):
    """Verifies anonymous requests may target a plain-HTTP endpoint (e.g. local)."""
    mock_post.return_value = _mock_response(SUCCESS_BODY)

    AnySearchTool(search_url="http://localhost:8080/v1/search").run(query="crewai")

    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "http://localhost:8080/v1/search"
