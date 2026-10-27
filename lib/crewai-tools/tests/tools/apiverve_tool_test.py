import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools.tools.apiverve_tool import apiverve_tool
from crewai_tools.tools.apiverve_tool.apiverve_tool import (
    ApiverveFindApisTool,
    ApiverveRunApiTool,
)


API_KEY = "test-apiverve-key"

CATALOG = {
    "apis": {
        "emailvalidator": {
            "title": "Email Validator",
            "category": "Validation",
            "description": "Validate an email address",
            "method": "GET",
            "path": "/v1/emailvalidator",
            "params": [
                {"name": "email", "type": "string", "required": True, "in": "query"}
            ],
        },
        "vatrates": {
            "title": "VAT Rates",
            "category": "Finance",
            "description": "VAT rates by country",
            "method": "GET",
            "path": "/v1/vatrates",
            "params": [],
        },
        "gravatarlookup": {
            "title": "Gravatar Lookup",
            "category": "Social",
            "description": "Look up a Gravatar profile",
            "method": "GET",
            "path": "/v1/gravatarlookup",
            "params": [],
        },
        "texttopdf": {
            "title": "Text to PDF",
            "category": "Documents",
            "description": "Make a PDF",
            "method": "POST",
            "path": "/v1/texttopdf",
            "params": [
                {"name": "text", "type": "string", "required": True, "in": "body"},
                {"name": "title", "type": "string", "in": "query"},
            ],
        },
    }
}


def _response(status_code=200, body=None):
    response = MagicMock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.json.return_value = body
    response.raise_for_status.side_effect = (
        requests.HTTPError(str(status_code)) if status_code >= 400 else None
    )
    return response


@pytest.fixture(autouse=True)
def reset_catalog_cache():
    apiverve_tool._catalog = None
    apiverve_tool._catalog_at = 0.0
    yield
    apiverve_tool._catalog = None


@pytest.fixture
def catalog_get():
    with patch.object(
        apiverve_tool.requests, "get", return_value=_response(body=CATALOG)
    ) as mock_get:
        yield mock_get


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("APIVERVE_API_KEY", API_KEY)


def test_find_ranks_word_starts_without_a_key(monkeypatch, catalog_get):
    monkeypatch.delenv("APIVERVE_API_KEY", raising=False)

    rows = json.loads(ApiverveFindApisTool().run(query="vat"))

    assert [row["api"] for row in rows] == ["vatrates"]
    assert "x-api-key" not in catalog_get.call_args.kwargs["headers"]


def test_find_returns_parameters(catalog_get):
    rows = json.loads(ApiverveFindApisTool().run(query="email validator"))

    assert rows[0]["api"] == "emailvalidator"
    assert rows[0]["parameters"][0] == {
        "name": "email",
        "type": "string",
        "required": True,
        "description": None,
        "example": None,
    }


def test_catalog_is_cached(catalog_get):
    tool = ApiverveFindApisTool()
    tool.run(query="vat")
    tool.run(query="pdf")

    assert catalog_get.call_count == 1


def test_malformed_catalog_is_an_error():
    with patch.object(
        apiverve_tool.requests, "get", return_value=_response(body={"apis": []})
    ):
        result = ApiverveFindApisTool().run(query="vat")

    assert result.startswith("Error: The APIVerve catalog could not be loaded")
    assert apiverve_tool._catalog is None


def test_failed_refresh_serves_stale_catalog_and_backs_off(catalog_get):
    tool = ApiverveFindApisTool()
    tool.run(query="vat")
    apiverve_tool._catalog_at -= apiverve_tool.CATALOG_TTL_SECONDS + 1

    with patch.object(
        apiverve_tool.requests, "get", side_effect=requests.ConnectionError("down")
    ) as failing_get:
        rows = json.loads(tool.run(query="vat"))
        tool.run(query="pdf")

    assert [row["api"] for row in rows] == ["vatrates"]
    assert failing_get.call_count == 1


def test_run_requires_env_var(monkeypatch):
    monkeypatch.delenv("APIVERVE_API_KEY", raising=False)

    result = ApiverveRunApiTool().run(api="emailvalidator", parameters={})

    assert "APIVERVE_API_KEY" in result


def test_run_get_sends_key_to_fixed_host(api_key, catalog_get):
    with patch.object(
        apiverve_tool.requests,
        "request",
        return_value=_response(body={"status": "ok", "data": {"isValid": True}}),
    ) as mock_request:
        result = ApiverveRunApiTool().run(
            api="emailvalidator", parameters={"email": "jane@example.com"}
        )

    assert json.loads(result) == {"isValid": True}
    args, kwargs = mock_request.call_args
    assert args == ("GET", "https://api.apiverve.com/v1/emailvalidator")
    assert kwargs["params"] == {"email": "jane@example.com"}
    assert kwargs["json"] is None
    assert kwargs["headers"]["x-api-key"] == API_KEY
    assert kwargs["allow_redirects"] is False


def test_run_post_places_query_and_body_per_catalog(api_key, catalog_get):
    with patch.object(
        apiverve_tool.requests,
        "request",
        return_value=_response(body={"status": "ok", "data": {"pdf": "x"}}),
    ) as mock_request:
        result = ApiverveRunApiTool().run(
            api="texttopdf", parameters={"text": "hi", "title": "T"}
        )

    assert json.loads(result) == {"pdf": "x"}
    args, kwargs = mock_request.call_args
    assert args == ("POST", "https://api.apiverve.com/v1/texttopdf")
    assert kwargs["params"] == {"title": "T"}
    assert kwargs["json"] == {"text": "hi"}


def test_run_reports_missing_required_parameter(api_key, catalog_get):
    result = ApiverveRunApiTool().run(api="emailvalidator", parameters={})

    assert result == "Error: emailvalidator needs: email."


def test_run_reports_unknown_api(api_key, catalog_get):
    result = ApiverveRunApiTool().run(api="nope", parameters={})

    assert result.startswith('Error: Unknown APIVerve API "nope"')


def test_run_reports_bad_key(api_key, catalog_get):
    with patch.object(
        apiverve_tool.requests,
        "request",
        return_value=_response(401, {"status": "error", "error": "Unauthorized"}),
    ):
        result = ApiverveRunApiTool().run(
            api="emailvalidator", parameters={"email": "jane@example.com"}
        )

    assert result == "Error: APIVerve rejected the API key."


def test_run_never_echoes_the_key(api_key, catalog_get):
    body = {"status": "ok", "data": {"note": f"called with {API_KEY}"}}
    error = {"status": "error", "error": f"bad key {API_KEY}"}
    tool = ApiverveRunApiTool()

    with patch.object(
        apiverve_tool.requests, "request", return_value=_response(body=body)
    ):
        ok = tool.run(api="emailvalidator", parameters={"email": "jane@example.com"})
    with patch.object(
        apiverve_tool.requests, "request", return_value=_response(400, error)
    ):
        failed = tool.run(
            api="emailvalidator", parameters={"email": "jane@example.com"}
        )

    for result in (ok, failed):
        assert API_KEY not in result
        assert "[redacted]" in result
