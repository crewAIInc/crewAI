from unittest.mock import Mock, patch

import pytest
import requests

from crewai_tools import (
    CompanyEnrichCompanyTool,
    CompanyEnrichSearchTool,
    CompanyEnrichSimilarTool,
)


@pytest.fixture
def mock_request(monkeypatch):
    monkeypatch.setenv("COMPANYENRICH_API_KEY", "test-companyenrich-key")
    with patch(
        "crewai_tools.tools.companyenrich_tool.companyenrich_tool.requests.request"
    ) as mock:
        mock.return_value.status_code = 200
        mock.return_value.json.return_value = {"items": [], "page": 2, "totalPages": 3}
        yield mock


def test_enrich_domain_and_explicit_credentials(mock_request):
    tool = CompanyEnrichCompanyTool(api_key="explicit-test-key", timeout=12)
    mock_request.return_value.json.return_value = {
        "name": "Example",
        "domain": "example.com",
    }
    assert tool.run(domain=" example.com ") == {
        "name": "Example",
        "domain": "example.com",
    }
    mock_request.assert_called_once_with(
        "GET",
        "https://api.companyenrich.com/companies/enrich",
        headers={
            "Authorization": "Bearer explicit-test-key",
            "Accept": "application/json",
        },
        params={"domain": "example.com"},
        json=None,
        timeout=12,
        allow_redirects=False,
    )
    assert "explicit-test-key" not in repr(tool)
    assert "api_key" not in tool.model_dump()
    assert "api_key" not in tool.args_schema.model_json_schema()["properties"]


@pytest.mark.parametrize(
    "tool,arguments,endpoint,payload",
    [
        (
            CompanyEnrichSearchTool,
            {"semantic_query": "B2B software", "page": 2, "page_size": 3},
            "/companies/search",
            {"semanticQuery": "B2B software", "page": 2, "pageSize": 3},
        ),
        (
            CompanyEnrichSimilarTool,
            {"domains": ["stripe.com"], "page": 2, "page_size": 3},
            "/companies/similar",
            {"domains": ["stripe.com"], "page": 2, "pageSize": 3},
        ),
    ],
)
def test_discovery_body_and_pagination_metadata(
    mock_request, tool, arguments, endpoint, payload
):
    assert tool().run(**arguments) == {"items": [], "page": 2, "totalPages": 3}
    mock_request.assert_called_once_with(
        "POST",
        f"https://api.companyenrich.com{endpoint}",
        headers={
            "Authorization": "Bearer test-companyenrich-key",
            "Accept": "application/json",
        },
        params=None,
        json=payload,
        timeout=30,
        allow_redirects=False,
    )


@pytest.mark.parametrize(
    "tool,arguments",
    [
        (CompanyEnrichCompanyTool, {"domain": " "}),
        (CompanyEnrichCompanyTool, {"domain": "a" * 1025}),
        (CompanyEnrichSearchTool, {"semantic_query": "software", "page_size": 101}),
        (CompanyEnrichSearchTool, {"semantic_query": "software", "page": 0}),
        (CompanyEnrichSearchTool, {"semantic_query": " "}),
        (CompanyEnrichSearchTool, {"semantic_query": "a" * 501}),
        (CompanyEnrichSimilarTool, {"domains": []}),
        (CompanyEnrichSimilarTool, {"domains": ["example.com"] * 11}),
        (CompanyEnrichSimilarTool, {"domains": [" "]}),
    ],
)
def test_invalid_arguments_do_not_make_requests(mock_request, tool, arguments):
    with pytest.raises(ValueError, match="arguments validation failed"):
        tool().run(**arguments)
    mock_request.assert_not_called()


def test_search_result_window(mock_request):
    with pytest.raises(ValueError, match="10000"):
        CompanyEnrichSearchTool().run(
            semantic_query="software", page=101, page_size=100
        )
    mock_request.assert_not_called()


def test_missing_credentials(mock_request, monkeypatch):
    monkeypatch.delenv("COMPANYENRICH_API_KEY")
    with pytest.raises(ValueError, match="COMPANYENRICH_API_KEY"):
        CompanyEnrichCompanyTool().run(domain="example.com")
    mock_request.assert_not_called()


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_http_errors_are_not_retried(mock_request, status):
    mock_request.return_value.raise_for_status.side_effect = requests.HTTPError(
        response=Mock(status_code=status)
    )
    with pytest.raises(requests.HTTPError):
        CompanyEnrichCompanyTool().run(domain="example.com")
    mock_request.assert_called_once()


def test_timeout_is_not_retried(mock_request):
    mock_request.side_effect = requests.Timeout()
    with pytest.raises(requests.Timeout):
        CompanyEnrichSearchTool().run(semantic_query="software")
    mock_request.assert_called_once()


def test_unexpected_response(mock_request):
    mock_request.return_value.json.return_value = []
    with pytest.raises(TypeError, match="JSON object"):
        CompanyEnrichCompanyTool().run(domain="example.com")


def test_redirect_is_not_followed(mock_request):
    mock_request.return_value.status_code = 302
    with pytest.raises(ValueError, match="redirect"):
        CompanyEnrichCompanyTool().run(domain="example.com")
    assert mock_request.call_args.kwargs["allow_redirects"] is False
