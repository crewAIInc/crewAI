import json
from unittest.mock import MagicMock, patch

from crewai_tools import FXMacroDataTool
import pytest
import requests


PAYLOAD = json.dumps({"currency": "USD", "data": []})
SAFE_GET = "crewai_tools.tools.fxmacrodata_tool.fxmacrodata_tool.safe_get"


@pytest.fixture
def tool():
    return FXMacroDataTool(api_key="test_key")


def _response(body=PAYLOAD, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.text = body
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *args: None
    return response


def _mock_safe_get(captured, body=PAYLOAD, status_code=200):
    """Return a safe_get stand-in that records what it was asked to fetch."""

    def _get(url, *, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["headers"] = dict(headers or {})
        captured["timeout"] = timeout
        return _response(body, status_code)

    return _get


def test_catalogue_builds_the_discovery_endpoint(tool):
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured)):
        result = tool.run(dataset="catalogue", currency="USD")

    assert captured["url"] == "https://api.fxmacrodata.com/v1/data_catalogue/usd"
    assert captured["timeout"] == FXMacroDataTool.REQUEST_TIMEOUT
    assert result == PAYLOAD


def test_latest_snapshot_needs_no_indicator(tool):
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured)):
        tool.run(dataset="latest", currency="JPY")

    assert captured["url"] == "https://api.fxmacrodata.com/v1/announcements/jpy/latest"


def test_history_passes_the_window_through(tool):
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured)):
        tool.run(
            dataset="history",
            currency="USD",
            indicator="inflation",
            start_date="2024-01-01",
            limit=5,
        )

    assert "announcements/usd/inflation" in captured["url"]
    assert "start_date=2024-01-01" in captured["url"]
    assert "limit=5" in captured["url"]
    # end_date was not supplied, so it must not be sent at all.
    assert "end_date" not in captured["url"]


def test_history_without_an_indicator_points_at_the_catalogue(tool):
    result = tool.run(dataset="history", currency="USD")

    assert "needs an indicator slug" in result
    assert "catalogue" in result


def test_pair_datasets_require_both_sides(tool):
    result = tool.run(dataset="fx_rate", base="EUR")

    assert "needs both base and quote" in result


def test_pair_datasets_build_both_sides(tool):
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured)):
        tool.run(dataset="rate_differential", base="USD", quote="JPY")

    assert captured["url"].startswith(
        "https://api.fxmacrodata.com/v1/rate_differentials/usd/jpy"
    )


def test_api_key_travels_as_a_header_not_a_query_parameter(tool):
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured)):
        tool.run(dataset="market_sessions")

    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["x-api-key"] == "test_key"
    assert "test_key" not in captured["url"]


def test_no_auth_header_without_a_key(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured)):
        FXMacroDataTool().run(dataset="risk_sentiment")

    headers = {key.lower() for key in captured["headers"]}
    assert "x-api-key" not in headers


def test_key_is_never_sent_over_plain_http():
    tool = FXMacroDataTool(api_key="test_key", base_url="http://example.com/v1")
    with patch(SAFE_GET) as safe_get:
        result = tool.run(dataset="market_sessions")

    safe_get.assert_not_called()
    assert "non-HTTPS" in result
    assert "test_key" not in result


def test_key_from_the_environment_is_never_sent_over_plain_http(monkeypatch):
    monkeypatch.setenv("FXMACRODATA_API_KEY", "env_key")
    tool = FXMacroDataTool(base_url="http://example.com/v1")
    with patch(SAFE_GET) as safe_get:
        result = tool.run(dataset="market_sessions")

    safe_get.assert_not_called()
    assert "non-HTTPS" in result
    assert "env_key" not in result


def test_anonymous_reads_still_work_over_plain_http(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    captured = {}
    tool = FXMacroDataTool(base_url="http://example.com/v1")
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured)):
        result = tool.run(dataset="market_sessions")

    assert captured["url"] == "http://example.com/v1/market_sessions"
    assert "x-api-key" not in {key.lower() for key in captured["headers"]}
    assert result == PAYLOAD


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("indicator", "../../../internal/admin"),
        ("indicator", "usd?limit=100000&/latest"),
        ("indicator", "usd#/latest"),
        ("currency", "usd/../admin"),
        ("currency", "USDX"),
        ("currency", "us"),
    ],
)
def test_path_segments_that_could_escape_the_endpoint_are_rejected(tool, field, value):
    kwargs = {"dataset": "history", "indicator": "inflation", field: value}
    with patch(SAFE_GET) as safe_get:
        result = tool._run(**kwargs)

    safe_get.assert_not_called()
    assert "Invalid arguments" in result


def test_pair_currencies_are_validated_like_the_currency_field(tool):
    with patch(SAFE_GET) as safe_get:
        result = tool._run(dataset="fx_rate", base="USD", quote="jpy/../../admin")

    safe_get.assert_not_called()
    assert "Invalid arguments" in result


def test_path_segments_are_percent_encoded():
    # The schema already rejects these characters; encoding is the second line
    # of defence for anything that reaches _resolve by another route.
    assert FXMacroDataTool._segment("a/b?c#d") == "a%2Fb%3Fc%23d"
    assert FXMacroDataTool._segment("non_farm_payrolls") == "non_farm_payrolls"


def test_refused_base_url_is_reported_not_raised(tool):
    refusal = ValueError("URL resolves to private/reserved IP 127.0.0.1")
    with patch(SAFE_GET, side_effect=refusal):
        result = tool.run(dataset="market_sessions")

    assert "was refused" in result
    assert "private/reserved IP" in result


def test_auth_failure_explains_the_key_requirement(tool):
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured, status_code=403)):
        result = tool.run(dataset="cot", currency="GBP")

    assert "requires an API key" in result
    assert "USD" in result


def test_server_error_is_not_reported_as_an_auth_problem(tool):
    captured = {}
    with patch(SAFE_GET, side_effect=_mock_safe_get(captured, status_code=500)):
        result = tool.run(dataset="commodities")

    assert "HTTP 500" in result
    assert "API key" not in result


def test_network_failure_is_returned_not_raised(tool):
    with patch(SAFE_GET, side_effect=requests.ConnectionError("boom")):
        result = tool.run(dataset="market_sessions")

    assert "market_sessions" in result
    assert "failed" in result
    # Connection errors are OSErrors, whose message is replaced by the class
    # name so OS-added path context never leaks into the agent's output.
    assert "ConnectionError" in result


def test_invalid_json_is_reported(tool):
    captured = {}
    with patch(
        SAFE_GET, side_effect=_mock_safe_get(captured, body="<html>not json</html>")
    ):
        result = tool.run(dataset="market_sessions")

    assert "not valid JSON" in result


def test_unknown_dataset_is_rejected_by_the_schema(tool):
    # BaseTool.run validates against args_schema before _run is reached, so an
    # unknown dataset never becomes a request. The Literal is what constrains it.
    with pytest.raises(Exception) as excinfo:
        tool.run(dataset="not_a_dataset")

    assert "dataset" in str(excinfo.value)


def test_direct_run_call_still_reports_bad_arguments(tool):
    # _run can be called directly, bypassing the schema, so it validates again
    # and returns a message rather than raising into the agent loop.
    result = tool._run(dataset="not_a_dataset")

    assert "Invalid arguments" in result
