import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests

from crewai_tools.tools.zenrows_tool.zenrows_scrape_tool import ZenRowsScrapeTool


TOOL_MODULE = "crewai_tools.tools.zenrows_tool.zenrows_scrape_tool"


def _mock_response(
    status_code: int = 200,
    text: str = "# Example\n\nHello world",
    json_body: dict | None = None,
):
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    response.json = Mock(
        return_value=json_body if json_body is not None else {},
        side_effect=None if json_body is not None else ValueError("no json"),
    )
    if status_code >= 400:
        error = requests.HTTPError(response=response)
        response.raise_for_status = Mock(side_effect=error)
    else:
        response.raise_for_status = Mock()
    return response


@patch.dict("os.environ", {}, clear=True)
def test_requires_api_key():
    with pytest.raises(ValueError, match="ZENROWS_API_KEY"):
        ZenRowsScrapeTool()


@patch.dict("os.environ", {}, clear=True)
def test_explicit_api_key_is_accepted_without_env_var():
    tool = ZenRowsScrapeTool(api_key="explicit-key")
    assert tool._resolved_api_key == "explicit-key"


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_default_config_is_adaptive_stealth():
    tool = ZenRowsScrapeTool()
    assert tool.config == {"mode": "auto"}


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_proxy_country_without_premium_proxy_is_rejected():
    with pytest.raises(ValueError, match="premium_proxy"):
        ZenRowsScrapeTool(config={"mode": "auto", "proxy_country": "us"})

    # Also rejected outside Adaptive Stealth -- proxy_country alone is a
    # no-op regardless of `mode`.
    with pytest.raises(ValueError, match="premium_proxy"):
        ZenRowsScrapeTool(config={"proxy_country": "us"})


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_premium_proxy_or_js_render_with_mode_auto_is_rejected():
    with pytest.raises(ValueError, match='mode: "auto"'):
        ZenRowsScrapeTool(config={"mode": "auto", "premium_proxy": True})

    with pytest.raises(ValueError, match='mode: "auto"'):
        ZenRowsScrapeTool(config={"mode": "auto", "js_render": True})


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_premium_proxy_and_proxy_country_without_mode_auto_is_accepted():
    tool = ZenRowsScrapeTool(config={"premium_proxy": True, "proxy_country": "us"})
    assert tool.config == {"premium_proxy": True, "proxy_country": "us"}


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_run_success_sends_url_and_apikey(mock_get):
    mock_get.return_value = _mock_response(text="# Hello")
    tool = ZenRowsScrapeTool()

    result = tool._run(url="https://example.com")

    assert result == "# Hello"
    called_url, called_kwargs = mock_get.call_args
    assert called_url[0] == "https://api.zenrows.com/v1/"
    params = called_kwargs["params"]
    assert params["url"] == "https://example.com"
    assert params["apikey"] == "test_api_key"
    assert params["mode"] == "auto"
    # Default response_type ("markdown") is sent explicitly.
    assert params["response_type"] == "markdown"


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_html_response_type_omits_the_param(mock_get):
    mock_get.return_value = _mock_response(text="<html></html>")
    tool = ZenRowsScrapeTool()

    tool._run(url="https://example.com", response_type="html")

    params = mock_get.call_args.kwargs["params"]
    assert "response_type" not in params


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_html_response_type_overrides_one_set_via_config(mock_get):
    """`config` isn't documented as a place to set `response_type`, but if a
    caller puts one there anyway, an explicit per-call "html" must still win
    rather than silently returning the configured format.
    """
    mock_get.return_value = _mock_response(text="<html></html>")
    tool = ZenRowsScrapeTool(config={"response_type": "plaintext"})

    tool._run(url="https://example.com", response_type="html")

    params = mock_get.call_args.kwargs["params"]
    assert "response_type" not in params


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_invalid_response_type_is_rejected_before_any_request():
    tool = ZenRowsScrapeTool()
    with patch(f"{TOOL_MODULE}.requests.get") as mock_get:
        with pytest.raises(ValueError, match="Unsupported response_type"):
            tool._run(url="https://example.com", response_type="pdf")
    mock_get.assert_not_called()


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_unsafe_url_is_rejected_before_any_request():
    tool = ZenRowsScrapeTool()
    with patch(f"{TOOL_MODULE}.requests.get") as mock_get:
        with pytest.raises(ValueError):
            tool._run(url="file:///etc/passwd")
    mock_get.assert_not_called()


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_custom_config_is_merged_into_every_request(mock_get):
    mock_get.return_value = _mock_response()
    tool = ZenRowsScrapeTool(
        config={"js_render": True, "premium_proxy": True, "proxy_country": "us"}
    )

    tool._run(url="https://example.com")

    params = mock_get.call_args.kwargs["params"]
    assert params["js_render"] is True
    assert params["premium_proxy"] is True
    assert params["proxy_country"] == "us"
    # Adaptive Stealth's "mode" key is not force-added when the caller
    # supplies their own config.
    assert "mode" not in params


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_config_cannot_override_the_resolved_api_key(mock_get):
    mock_get.return_value = _mock_response()
    tool = ZenRowsScrapeTool(config={"apikey": "attacker-supplied-key"})

    tool._run(url="https://example.com")

    params = mock_get.call_args.kwargs["params"]
    assert params["apikey"] == "test_api_key"


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_timeout_returns_actionable_message_instead_of_raising(mock_get):
    mock_get.side_effect = requests.Timeout()
    tool = ZenRowsScrapeTool()

    result = tool._run(url="https://example.com")

    assert "timed out" in result
    assert "https://example.com" in result


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_401_reports_invalid_api_key(mock_get):
    mock_get.return_value = _mock_response(
        status_code=401, json_body={"message": "Invalid API key", "code": "AUTH001"}
    )
    tool = ZenRowsScrapeTool()

    result = tool._run(url="https://example.com")

    assert "401" in result
    assert "invalid or revoked API key" in result
    assert "Invalid API key" in result
    assert "AUTH001" in result


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_402_reports_plan_or_credit_issue(mock_get):
    mock_get.return_value = _mock_response(
        status_code=402, json_body={"message": "Domain not enabled", "code": "AUTH010"}
    )
    tool = ZenRowsScrapeTool()

    result = tool._run(url="https://example.com")

    assert "402" in result
    assert "plan/credit limit" in result
    assert "AUTH010" in result


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_429_reports_rate_limit(mock_get):
    mock_get.return_value = _mock_response(status_code=429, json_body={})
    tool = ZenRowsScrapeTool()

    result = tool._run(url="https://example.com")

    assert "429" in result
    assert "Retry after a short delay" in result


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_generic_request_exception_is_returned_not_raised(mock_get):
    mock_get.side_effect = requests.ConnectionError("connection refused")
    tool = ZenRowsScrapeTool()

    result = tool._run(url="https://example.com")

    assert "Zenrows request failed" in result
    assert "connection refused" in result


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
@patch(f"{TOOL_MODULE}.requests.get")
def test_arun_delegates_to_run(mock_get):
    mock_get.return_value = _mock_response(text="ok")
    tool = ZenRowsScrapeTool()

    result = asyncio.run(tool._arun(url="https://example.com"))

    assert result == "ok"


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_arun_offloads_the_blocking_call_to_a_worker_thread():
    """`_run` makes a blocking `requests.get()` call; `_arun` must not run it
    directly on the event loop, or one slow scrape would stall every other
    concurrent task in an async crew.
    """
    tool = ZenRowsScrapeTool()
    with patch(f"{TOOL_MODULE}.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = "ok"
        result = asyncio.run(tool._arun(url="https://example.com"))

    assert result == "ok"
    mock_to_thread.assert_called_once_with(tool._run, url="https://example.com")


@patch.dict("os.environ", {"ZENROWS_API_KEY": "test_api_key"})
def test_api_key_is_not_a_public_model_field_value():
    """The resolved credential must not round-trip through model_dump()."""
    tool = ZenRowsScrapeTool()
    dumped = tool.model_dump()
    assert "test_api_key" not in str(dumped)
