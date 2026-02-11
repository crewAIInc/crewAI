import os
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from crewai_tools.tools.brave_search_tool.base import BraveSearchToolBase
from crewai_tools.tools.brave_search_tool.brave_web_tool import BraveWebSearchTool
from crewai_tools.tools.brave_search_tool.brave_image_tool import BraveImageSearchTool
from crewai_tools.tools.brave_search_tool.brave_news_tool import BraveNewsSearchTool
from crewai_tools.tools.brave_search_tool.brave_video_tool import BraveVideoSearchTool
from crewai_tools.tools.brave_search_tool.brave_llm_context_tool import (
    BraveLLMContextTool,
)
from crewai_tools.tools.brave_search_tool.brave_local_pois_tool import (
    BraveLocalPOIsTool,
    BraveLocalPOIsDescriptionTool,
)
from crewai_tools.tools.brave_search_tool.schemas import (
    WebSearchParams,
    WebSearchHeaders,
    ImageSearchParams,
    ImageSearchHeaders,
    NewsSearchParams,
    NewsSearchHeaders,
    VideoSearchParams,
    VideoSearchHeaders,
    LLMContextParams,
    LLMContextHeaders,
    LocalPOIsParams,
    LocalPOIsHeaders,
    LocalPOIsDescriptionParams,
    LocalPOIsDescriptionHeaders,
)


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
    text: str = "",
) -> MagicMock:
    """Build a ``requests.Response``-like mock with the attributes used by ``_make_request``."""
    resp = MagicMock(spec=requests_lib.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.url = "https://api.search.brave.com/res/v1/web/search?q=test"
    resp.text = text or (str(json_data) if json_data else "")
    resp.headers = headers or {}
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


# Fixtures


@pytest.fixture(autouse=True)
def _brave_env_and_rate_limit():
    """Set BRAVE_API_KEY for every test and reset the shared rate-limit clock."""
    BraveSearchToolBase._last_request_time = 0
    with patch.dict(os.environ, {"BRAVE_API_KEY": "test-api-key"}):
        yield


@pytest.fixture
def web_tool():
    return BraveWebSearchTool()


@pytest.fixture
def image_tool():
    return BraveImageSearchTool()


@pytest.fixture
def news_tool():
    return BraveNewsSearchTool()


@pytest.fixture
def video_tool():
    return BraveVideoSearchTool()


# Initialization

ALL_TOOL_CLASSES = [
    BraveWebSearchTool,
    BraveImageSearchTool,
    BraveNewsSearchTool,
    BraveVideoSearchTool,
    BraveLLMContextTool,
    BraveLocalPOIsTool,
    BraveLocalPOIsDescriptionTool,
]


@pytest.mark.parametrize("tool_cls", ALL_TOOL_CLASSES)
def test_instantiation_with_env_var(tool_cls):
    """Each tool can be created when BRAVE_API_KEY is in the environment."""
    tool = tool_cls()
    assert tool.api_key == "test-api-key"


@pytest.mark.parametrize("tool_cls", ALL_TOOL_CLASSES)
def test_instantiation_with_explicit_key(tool_cls):
    """An explicit api_key takes precedence over the environment."""
    tool = tool_cls(api_key="explicit-key")
    assert tool.api_key == "explicit-key"


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            BraveWebSearchTool()


def test_default_attributes():
    tool = BraveWebSearchTool()
    assert tool.save_file is False
    assert tool.n_results == 10
    assert tool._timeout == 30
    assert tool._requests_per_second == 1.0
    assert tool.raw is False


def test_custom_constructor_args():
    tool = BraveWebSearchTool(
        save_file=True,
        timeout=60,
        n_results=5,
        requests_per_second=0.5,
        raw=True,
    )
    assert tool.save_file is True
    assert tool._timeout == 60
    assert tool.n_results == 5
    assert tool._requests_per_second == 0.5
    assert tool.raw is True


# Headers


def test_default_headers():
    tool = BraveWebSearchTool()
    assert tool.headers["x-subscription-token"] == "test-api-key"
    assert tool.headers["accept"] == "application/json"


def test_set_headers_merges_and_normalizes():
    tool = BraveWebSearchTool()
    tool.set_headers({"Cache-Control": "no-cache"})
    assert tool.headers["cache-control"] == "no-cache"
    assert tool.headers["x-subscription-token"] == "test-api-key"


def test_set_headers_returns_self_for_chaining():
    tool = BraveWebSearchTool()
    assert tool.set_headers({"Cache-Control": "no-cache"}) is tool


def test_invalid_header_value_raises():
    tool = BraveImageSearchTool()
    with pytest.raises(ValueError, match="Invalid headers"):
        tool.set_headers({"Accept": "text/xml"})


# Endpoint & Schema Wiring


@pytest.mark.parametrize(
    "tool_cls, expected_url, expected_params, expected_headers",
    [
        (
            BraveWebSearchTool,
            "https://api.search.brave.com/res/v1/web/search",
            WebSearchParams,
            WebSearchHeaders,
        ),
        (
            BraveImageSearchTool,
            "https://api.search.brave.com/res/v1/images/search",
            ImageSearchParams,
            ImageSearchHeaders,
        ),
        (
            BraveNewsSearchTool,
            "https://api.search.brave.com/res/v1/news/search",
            NewsSearchParams,
            NewsSearchHeaders,
        ),
        (
            BraveVideoSearchTool,
            "https://api.search.brave.com/res/v1/videos/search",
            VideoSearchParams,
            VideoSearchHeaders,
        ),
        (
            BraveLLMContextTool,
            "https://api.search.brave.com/res/v1/llm/context",
            LLMContextParams,
            LLMContextHeaders,
        ),
        (
            BraveLocalPOIsTool,
            "https://api.search.brave.com/res/v1/local/pois",
            LocalPOIsParams,
            LocalPOIsHeaders,
        ),
        (
            BraveLocalPOIsDescriptionTool,
            "https://api.search.brave.com/res/v1/local/descriptions",
            LocalPOIsDescriptionParams,
            LocalPOIsDescriptionHeaders,
        ),
    ],
)
def test_tool_wiring(tool_cls, expected_url, expected_params, expected_headers):
    tool = tool_cls()
    assert tool.search_url == expected_url
    assert tool.args_schema is expected_params
    assert tool.header_schema is expected_headers


# Payload Refinement  (e.g., `query` -> `q`, `count` fallback, param pass-through)


def test_web_refine_request_payload_passes_all_params(web_tool):
    params = web_tool._common_payload_refinement(
        {
            "query": "test",
            "country": "US",
            "search_lang": "en",
            "count": 5,
            "offset": 2,
            "safesearch": "moderate",
            "freshness": "pw",
        }
    )
    refined_params = web_tool._refine_request_payload(params)

    assert refined_params["q"] == "test"
    assert "query" not in refined_params
    assert refined_params["count"] == 5
    assert refined_params["country"] == "US"
    assert refined_params["search_lang"] == "en"
    assert refined_params["offset"] == 2
    assert refined_params["safesearch"] == "moderate"
    assert refined_params["freshness"] == "pw"


def test_image_refine_request_payload_passes_all_params(image_tool):
    params = image_tool._common_payload_refinement(
        {
            "query": "cat photos",
            "country": "US",
            "search_lang": "en",
            "safesearch": "strict",
            "count": 50,
            "spellcheck": True,
        }
    )
    refined_params = image_tool._refine_request_payload(params)

    assert refined_params["q"] == "cat photos"
    assert "query" not in refined_params
    assert refined_params["country"] == "US"
    assert refined_params["safesearch"] == "strict"
    assert refined_params["count"] == 50
    assert refined_params["spellcheck"] is True


def test_news_refine_request_payload_passes_all_params(news_tool):
    params = news_tool._common_payload_refinement(
        {
            "query": "breaking news",
            "country": "US",
            "count": 10,
            "offset": 1,
            "freshness": "pd",
            "extra_snippets": True,
        }
    )
    refined_params = news_tool._refine_request_payload(params)

    assert refined_params["q"] == "breaking news"
    assert "query" not in refined_params
    assert refined_params["country"] == "US"
    assert refined_params["offset"] == 1
    assert refined_params["freshness"] == "pd"
    assert refined_params["extra_snippets"] is True


def test_video_refine_request_payload_passes_all_params(video_tool):
    params = video_tool._common_payload_refinement(
        {
            "query": "tutorial",
            "country": "US",
            "count": 25,
            "offset": 0,
            "safesearch": "strict",
            "freshness": "pm",
        }
    )
    refined_params = video_tool._refine_request_payload(params)

    assert refined_params["q"] == "tutorial"
    assert "query" not in refined_params
    assert refined_params["country"] == "US"
    assert refined_params["offset"] == 0
    assert refined_params["freshness"] == "pm"


def test_legacy_constructor_params_flow_into_query_params():
    """The legacy n_results and country constructor params are applied as defaults
    when count/country are not explicitly provided at call time."""
    tool = BraveWebSearchTool(n_results=3, country="BR")
    params = tool._common_payload_refinement({"query": "test"})

    assert params["count"] == 3
    assert params["country"] == "BR"


def test_legacy_constructor_params_do_not_override_explicit_query_params():
    """Explicit query-time count/country take precedence over constructor defaults."""
    tool = BraveWebSearchTool(n_results=3, country="BR")
    params = tool._common_payload_refinement(
        {"query": "test", "count": 10, "country": "US"}
    )

    assert params["count"] == 10
    assert params["country"] == "US"


def test_refine_request_payload_passes_multiple_goggles_as_multiple_params(web_tool):
    result = web_tool._refine_request_payload(
        {
            "query": "test",
            "goggles": ["goggle1", "goggle2"],
        }
    )
    assert result["goggles"] == ["goggle1", "goggle2"]


# Null-like / empty value stripping
#
# crewAI's ensure_all_properties_required (pydantic_schema_utils.py) marks
# every schema property as required for OpenAI strict-mode compatibility.
# Because optional Brave API parameters look required to the LLM, it fills
# them with placeholder junk — None, "", "null", or [].  The test below
# verifies that _common_payload_refinement strips these from optional fields.


def test_common_refinement_strips_null_like_values(web_tool):
    """_common_payload_refinement drops optional keys with None / '' / 'null' / []."""
    params = web_tool._common_payload_refinement(
        {
            "query": "test",
            "country": "US",
            "search_lang": "",
            "freshness": "null",
            "count": 5,
            "goggles": [],
        }
    )
    assert params["q"] == "test"
    assert params["country"] == "US"
    assert params["count"] == 5
    assert "search_lang" not in params
    assert "freshness" not in params
    assert "goggles" not in params


# End-to-End _run() with Mocked HTTP Response


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_web_search_end_to_end(mock_get, web_tool):
    web_tool.raw = True
    data = {"web": {"results": [{"title": "R", "url": "http://r.co"}]}}
    mock_get.return_value = _mock_response(json_data=data)

    result = web_tool._run(query="test")

    mock_get.assert_called_once()
    call_args = mock_get.call_args.kwargs
    assert call_args["params"]["q"] == "test"
    assert call_args["headers"]["x-subscription-token"] == "test-api-key"
    assert result == data


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_image_search_end_to_end(mock_get, image_tool):
    image_tool.raw = True
    data = {"results": [{"url": "http://img.co/a.jpg"}]}
    mock_get.return_value = _mock_response(json_data=data)

    assert image_tool._run(query="cats") == data


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_news_search_end_to_end(mock_get, news_tool):
    news_tool.raw = True
    data = {"results": [{"title": "News", "url": "http://n.co"}]}
    mock_get.return_value = _mock_response(json_data=data)

    assert news_tool._run(query="headlines") == data


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_video_search_end_to_end(mock_get, video_tool):
    video_tool.raw = True
    data = {"results": [{"title": "Vid", "url": "http://v.co"}]}
    mock_get.return_value = _mock_response(json_data=data)

    assert video_tool._run(query="python tutorial") == data


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_raw_false_calls_refine_response(mock_get, web_tool):
    """With raw=False (the default), _refine_response transforms the API response."""
    api_response = {
        "web": {
            "results": [
                {
                    "title": "CrewAI",
                    "url": "https://crewai.com",
                    "description": "AI agent framework",
                }
            ]
        }
    }
    mock_get.return_value = _mock_response(json_data=api_response)

    assert web_tool.raw is False
    result = web_tool._run(query="crewai")

    # The web tool's _refine_response extracts and reshapes results.
    # The key assertion: we should NOT get back the raw API envelope.
    assert result != api_response


# Backward Compatibility & Legacy Parameter Support


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_positional_query_argument(mock_get, web_tool):
    """tool.run('my query') works as a positional argument."""
    mock_get.return_value = _mock_response(json_data={})

    web_tool._run("positional test")

    assert mock_get.call_args.kwargs["params"]["q"] == "positional test"


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_search_query_backward_compat(mock_get, web_tool):
    """The legacy 'search_query' param is mapped to 'query'."""
    mock_get.return_value = _mock_response(json_data={})

    web_tool._run(search_query="legacy test")

    assert mock_get.call_args.kwargs["params"]["q"] == "legacy test"


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base._save_results_to_file")
def test_save_file_called_when_enabled(mock_save, mock_get):
    mock_get.return_value = _mock_response(json_data={"results": []})

    tool = BraveWebSearchTool(save_file=True)
    tool._run(query="test")

    mock_save.assert_called_once()


# Error Handling


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_connection_error_raises_runtime_error(mock_get, web_tool):
    mock_get.side_effect = requests_lib.exceptions.ConnectionError("refused")
    with pytest.raises(RuntimeError, match="Brave Search API connection failed"):
        web_tool._run(query="test")


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_timeout_raises_runtime_error(mock_get, web_tool):
    mock_get.side_effect = requests_lib.exceptions.Timeout("timed out")
    with pytest.raises(RuntimeError, match="timed out"):
        web_tool._run(query="test")


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_invalid_params_raises_value_error(mock_get, web_tool):
    """count=999 exceeds WebSearchParams.count le=20."""
    with pytest.raises(ValueError, match="Invalid parameters"):
        web_tool._run(query="test", count=999)


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_4xx_error_raises_with_api_detail(mock_get, web_tool):
    """A 422 with a structured error body includes code and detail in the message."""
    mock_get.return_value = _mock_response(
        status_code=422,
        json_data={
            "error": {
                "id": "abc-123",
                "status": 422,
                "code": "OPTION_NOT_IN_PLAN",
                "detail": "extra_snippets requires a Pro plan",
            }
        },
    )
    with pytest.raises(RuntimeError, match="OPTION_NOT_IN_PLAN") as exc_info:
        web_tool._run(query="test")
    assert "extra_snippets requires a Pro plan" in str(exc_info.value)
    assert "HTTP 422" in str(exc_info.value)


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_auth_error_raises_immediately(mock_get, web_tool):
    """A 401 with SUBSCRIPTION_TOKEN_INVALID is not retried."""
    mock_get.return_value = _mock_response(
        status_code=401,
        json_data={
            "error": {
                "id": "xyz",
                "status": 401,
                "code": "SUBSCRIPTION_TOKEN_INVALID",
                "detail": "The subscription token is invalid",
            }
        },
    )
    with pytest.raises(RuntimeError, match="SUBSCRIPTION_TOKEN_INVALID"):
        web_tool._run(query="test")
    # Should NOT have retried — only one call.
    assert mock_get.call_count == 1


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_quota_limited_429_raises_immediately(mock_get, web_tool):
    """A 429 with QUOTA_LIMITED is NOT retried — quota exhaustion is terminal."""
    mock_get.return_value = _mock_response(
        status_code=429,
        json_data={
            "error": {
                "id": "ql-1",
                "status": 429,
                "code": "QUOTA_LIMITED",
                "detail": "Monthly quota exceeded",
            }
        },
    )
    with pytest.raises(RuntimeError, match="QUOTA_LIMITED") as exc_info:
        web_tool._run(query="test")
    assert "Monthly quota exceeded" in str(exc_info.value)
    # Terminal — only one HTTP call, no retries.
    assert mock_get.call_count == 1


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_usage_limit_exceeded_429_raises_immediately(mock_get, web_tool):
    """USAGE_LIMIT_EXCEEDED is also non-retryable, just like QUOTA_LIMITED."""
    mock_get.return_value = _mock_response(
        status_code=429,
        json_data={
            "error": {
                "id": "ule-1",
                "status": 429,
                "code": "USAGE_LIMIT_EXCEEDED",
            }
        },
        text="usage limit exceeded",
    )
    with pytest.raises(RuntimeError, match="USAGE_LIMIT_EXCEEDED"):
        web_tool._run(query="test")
    assert mock_get.call_count == 1


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_error_body_is_fully_included_in_message(mock_get, web_tool):
    """The full JSON error body is included in the RuntimeError message."""
    mock_get.return_value = _mock_response(
        status_code=429,
        json_data={
            "error": {
                "id": "x",
                "status": 429,
                "code": "QUOTA_LIMITED",
                "detail": "Exceeded",
                "meta": {"plan": "free", "limit": 1000},
            }
        },
    )
    with pytest.raises(RuntimeError) as exc_info:
        web_tool._run(query="test")
    msg = str(exc_info.value)
    assert "HTTP 429" in msg
    assert "QUOTA_LIMITED" in msg
    assert "free" in msg
    assert "1000" in msg


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_error_without_json_body_falls_back_to_text(mock_get, web_tool):
    """When the error response isn't valid JSON, resp.text is used as the detail."""
    resp = _mock_response(status_code=500, text="Internal Server Error")
    resp.json.side_effect = ValueError("No JSON")
    mock_get.return_value = resp

    with pytest.raises(RuntimeError, match="Internal Server Error"):
        web_tool._run(query="test")


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
def test_invalid_json_on_success_raises_runtime_error(mock_get, web_tool):
    """A 200 OK with a non-JSON body raises RuntimeError."""
    resp = _mock_response(status_code=200)
    resp.json.side_effect = ValueError("Expecting value")
    mock_get.return_value = resp

    with pytest.raises(RuntimeError, match="invalid JSON"):
        web_tool._run(query="test")


# Rate Limiting


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_rate_limit_sleeps_when_too_fast(mock_time, mock_get, web_tool):
    """Back-to-back calls within the interval trigger a sleep."""
    mock_get.return_value = _mock_response(json_data={})

    # Simulate: last request was at t=100, "now" is t=100.2 (only 0.2s elapsed).
    # With default 1 req/s the min interval is 1.0s, so it should sleep ~0.8s.
    mock_time.time.return_value = 100.2
    BraveSearchToolBase._last_request_time = 100.0

    web_tool._run(query="test")

    mock_time.sleep.assert_called_once()
    sleep_duration = mock_time.sleep.call_args[0][0]
    assert 0.7 < sleep_duration < 0.9  # approximately 0.8s


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_rate_limit_skips_sleep_when_enough_time_passed(mock_time, mock_get, web_tool):
    """No sleep when the elapsed time already exceeds the interval."""
    mock_get.return_value = _mock_response(json_data={})

    # Last request was at t=100, "now" is t=102 (2s elapsed > 1s interval).
    mock_time.time.return_value = 102.0
    BraveSearchToolBase._last_request_time = 100.0

    web_tool._run(query="test")

    mock_time.sleep.assert_not_called()


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_rate_limit_disabled_when_zero(mock_time, mock_get):
    """requests_per_second=0 disables rate limiting entirely."""
    mock_get.return_value = _mock_response(json_data={})

    tool = BraveWebSearchTool(requests_per_second=0)
    BraveSearchToolBase._last_request_time = 100.0
    mock_time.time.return_value = 100.0  # same instant

    tool._run(query="test")

    mock_time.sleep.assert_not_called()


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_rate_limit_shared_across_instances(mock_time, mock_get):
    """The rate-limit clock is shared across different tool instances."""
    mock_get.return_value = _mock_response(json_data={})

    web_tool = BraveWebSearchTool()
    image_tool = BraveImageSearchTool()

    # Web tool fires at t=100, updating the class-level timestamp.
    mock_time.time.return_value = 100.0
    BraveSearchToolBase._last_request_time = 0
    web_tool._run(query="test")

    # Image tool fires at t=100.3 — within the 1s window set by web_tool.
    mock_time.time.return_value = 100.3
    image_tool._run(query="cats")

    mock_time.sleep.assert_called_once()
    sleep_duration = mock_time.sleep.call_args[0][0]
    assert 0.6 < sleep_duration < 0.8  # approximately 0.7s


# Retry Behavior


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_429_rate_limited_retries_then_succeeds(mock_time, mock_get, web_tool):
    """A transient RATE_LIMITED 429 is retried; success on the second attempt."""
    mock_time.time.return_value = 200.0

    resp_429 = _mock_response(
        status_code=429,
        json_data={"error": {"id": "r", "status": 429, "code": "RATE_LIMITED"}},
        headers={"Retry-After": "2"},
    )
    resp_200 = _mock_response(status_code=200, json_data={"web": {"results": []}})
    mock_get.side_effect = [resp_429, resp_200]

    web_tool.raw = True
    result = web_tool._run(query="test")

    assert result == {"web": {"results": []}}
    assert mock_get.call_count == 2
    # Slept for the Retry-After value.
    retry_sleeps = [c for c in mock_time.sleep.call_args_list if c[0][0] == 2.0]
    assert len(retry_sleeps) == 1


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_5xx_is_retried(mock_time, mock_get, web_tool):
    """A 502 server error is retried; success on the second attempt."""
    mock_time.time.return_value = 200.0

    resp_502 = _mock_response(status_code=502, text="Bad Gateway")
    resp_502.json.side_effect = ValueError("no json")
    resp_200 = _mock_response(status_code=200, json_data={"web": {"results": []}})
    mock_get.side_effect = [resp_502, resp_200]

    web_tool.raw = True
    result = web_tool._run(query="test")

    assert result == {"web": {"results": []}}
    assert mock_get.call_count == 2


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_429_rate_limited_exhausts_retries(mock_time, mock_get, web_tool):
    """Persistent RATE_LIMITED 429s exhaust retries and raise RuntimeError."""
    mock_time.time.return_value = 200.0

    resp_429 = _mock_response(
        status_code=429,
        json_data={"error": {"id": "r", "status": 429, "code": "RATE_LIMITED"}},
    )
    mock_get.return_value = resp_429

    with pytest.raises(RuntimeError, match="RATE_LIMITED"):
        web_tool._run(query="test")
    # 3 attempts (default _max_retries).
    assert mock_get.call_count == 3


@patch("crewai_tools.tools.brave_search_tool.base.requests.get")
@patch("crewai_tools.tools.brave_search_tool.base.time")
def test_retry_uses_exponential_backoff_when_no_retry_after(
    mock_time, mock_get, web_tool
):
    """Without Retry-After, backoff is 2^attempt (1s, 2s, ...)."""
    mock_time.time.return_value = 200.0

    resp_503 = _mock_response(status_code=503, text="Service Unavailable")
    resp_503.json.side_effect = ValueError("no json")
    resp_200 = _mock_response(status_code=200, json_data={"ok": True})
    mock_get.side_effect = [resp_503, resp_503, resp_200]

    web_tool.raw = True
    web_tool._run(query="test")

    # Two retries: attempt 0 → sleep(1.0), attempt 1 → sleep(2.0).
    retry_sleeps = [c[0][0] for c in mock_time.sleep.call_args_list]
    assert 1.0 in retry_sleeps
    assert 2.0 in retry_sleeps
