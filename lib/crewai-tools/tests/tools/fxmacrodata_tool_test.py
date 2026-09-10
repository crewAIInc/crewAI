import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from crewai_tools import FXMacroDataTool


PAYLOAD = json.dumps({"currency": "USD", "data": []})


@pytest.fixture
def tool():
    return FXMacroDataTool(api_key="test_key")


def _mock_urlopen(captured):
    """Return a urlopen stand-in that records the Request it was handed."""

    def _open(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["timeout"] = timeout
        response = MagicMock()
        response.read.return_value = PAYLOAD.encode("utf-8")
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *args: None
        return response

    return _open


def test_catalogue_builds_the_discovery_endpoint(tool):
    captured = {}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        result = tool.run(dataset="catalogue", currency="USD")

    assert captured["url"] == "https://api.fxmacrodata.com/v1/data_catalogue/usd"
    assert result == PAYLOAD


def test_latest_snapshot_needs_no_indicator(tool):
    captured = {}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        tool.run(dataset="latest", currency="JPY")

    assert captured["url"] == "https://api.fxmacrodata.com/v1/announcements/jpy/latest"


def test_history_passes_the_window_through(tool):
    captured = {}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
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
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        tool.run(dataset="rate_differential", base="USD", quote="JPY")

    assert captured["url"].startswith(
        "https://api.fxmacrodata.com/v1/rate_differentials/usd/jpy"
    )


def test_api_key_travels_as_a_header_not_a_query_parameter(tool):
    captured = {}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        tool.run(dataset="market_sessions")

    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["X-api-key".lower()] == "test_key"
    assert "test_key" not in captured["url"]


def test_no_auth_header_without_a_key(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    captured = {}
    with patch("urllib.request.urlopen", side_effect=_mock_urlopen(captured)):
        FXMacroDataTool().run(dataset="risk_sentiment")

    headers = {key.lower() for key in captured["headers"]}
    assert "x-api-key" not in headers


def test_auth_failure_explains_the_key_requirement(tool):
    error = urllib.error.HTTPError("url", 403, "Forbidden", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        result = tool.run(dataset="cot", currency="GBP")

    assert "requires an API key" in result
    assert "USD" in result


def test_server_error_is_not_reported_as_an_auth_problem(tool):
    error = urllib.error.HTTPError("url", 500, "Server Error", {}, None)
    with patch("urllib.request.urlopen", side_effect=error):
        result = tool.run(dataset="commodities")

    assert "HTTP 500" in result
    assert "API key" not in result


def test_network_failure_is_returned_not_raised(tool):
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        result = tool.run(dataset="market_sessions")

    assert "failed" in result


def test_invalid_json_is_reported(tool):
    def _open(request, timeout=None):
        response = MagicMock()
        response.read.return_value = b"<html>not json</html>"
        response.__enter__ = lambda self: self
        response.__exit__ = lambda self, *args: None
        return response

    with patch("urllib.request.urlopen", side_effect=_open):
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
