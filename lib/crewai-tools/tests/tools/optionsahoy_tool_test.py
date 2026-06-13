import json
from unittest.mock import MagicMock, patch

from crewai_tools.tools.optionsahoy_tool.optionsahoy_tool import OptionsAhoyTool
import pytest
import requests


@pytest.fixture
def tool():
    return OptionsAhoyTool()


def test_optionsahoy_tool_initialization():
    instance = OptionsAhoyTool()
    assert instance.name == "OptionsAhoy Equity Compensation Tax Calculator"
    assert instance.base_url == "https://optionsahoy.com"
    assert instance.timeout == 30


def test_optionsahoy_tool_custom_initialization():
    instance = OptionsAhoyTool(base_url="https://example.test/", timeout=60)
    assert instance.base_url == "https://example.test/"
    assert instance.timeout == 60


@patch("requests.post")
def test_optionsahoy_tool_run_posts_to_correct_endpoint(mock_post, tool):
    mock_response = MagicMock()
    mock_response.json.return_value = {"eligible": True, "exclusion": 2000000}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    inputs = {
        "acquisitionDate": "2018-01-01",
        "saleDate": "2026-02-01",
        "entityType": "us-c-corp",
        "acquisitionMethod": "original-issuance",
        "assetCategory": "under-50m",
        "industry": "tech-software",
        "activeBusiness": "yes",
        "adjustedBasis": 10000,
        "expectedGain": 2000000,
        "stateCode": "CA",
        "ordinaryIncome": 250000,
        "filingStatus": "single",
    }

    result = tool.run(calculator="qsbs", inputs=inputs)

    called_url = mock_post.call_args.args[0]
    assert called_url == "https://optionsahoy.com/api/v1/qsbs"
    assert mock_post.call_args.kwargs["json"] == inputs
    assert mock_post.call_args.kwargs["timeout"] == 30

    parsed = json.loads(result)
    assert parsed["eligible"] is True
    assert parsed["exclusion"] == 2000000


@patch("requests.post")
def test_optionsahoy_tool_strips_none_but_keeps_termination_date(mock_post, tool):
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    inputs = {
        "shares": 1000,
        "strike": 1.0,
        "fmv": 10.0,
        "filingStatus": "single",
        "ordinaryIncome": 200000,
        "stateCode": "CA",
        "carryforwardCredit": 0,
        "horizon": 5,
        "cashReturnRate": 0.04,
        "grantDate": "2022-01-01",
        "hasLeftCompany": False,
        "terminationDate": None,
        "ticker": None,
    }

    tool.run(calculator="amt-iso", inputs=inputs)

    sent = mock_post.call_args.kwargs["json"]
    assert "terminationDate" in sent
    assert sent["terminationDate"] is None
    assert "ticker" not in sent


def test_optionsahoy_tool_missing_required_field_raises(tool):
    with pytest.raises(ValueError, match="missing required input field"):
        tool.run(calculator="qsbs", inputs={"stateCode": "CA"})


@patch("requests.post")
def test_optionsahoy_tool_http_error_returns_error_json(mock_post, tool):
    error_response = MagicMock()
    error_response.status_code = 400
    error_response.json.return_value = {"error": "bad input"}
    http_error = requests.exceptions.HTTPError(response=error_response)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = http_error
    mock_post.return_value = mock_response

    inputs = {
        "positionValue": 100000,
        "sector": "technology",
        "protectionLevel": 0.9,
        "tenorYears": 1.0,
    }

    result = tool.run(calculator="protective-put", inputs=inputs)
    parsed = json.loads(result)
    assert "error" in parsed
    assert "bad input" in parsed["error"]


@patch("requests.post")
def test_optionsahoy_tool_request_exception_returns_error_json(mock_post, tool):
    mock_post.side_effect = requests.exceptions.ConnectionError("boom")

    inputs = {
        "positionValue": 100000,
        "sector": "technology",
        "protectionLevel": 0.9,
        "tenorYears": 1.0,
    }

    result = tool.run(calculator="protective-put", inputs=inputs)
    parsed = json.loads(result)
    assert "error" in parsed
    assert "boom" in parsed["error"]
