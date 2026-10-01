import json
import os
from unittest.mock import MagicMock, patch

from crewai.tools.tool_failure import ToolFailure
from crewai_tools import (
    NeuralVergeAmazonOfferTool,
    NeuralVergeAmazonProductSearchTool,
    NeuralVergeAmazonProductTool,
    NeuralVergeAmazonSellerProductsTool,
    NeuralVergeAmazonSellerTool,
    NeuralVergeCompanyFundingTool,
    NeuralVergeEmailFinderTool,
    NeuralVergeEmailValidationTool,
    NeuralVergeExtractTool,
    NeuralVergeLinkedInCompanyEmployeesTool,
    NeuralVergeLinkedInCompanySearchTool,
    NeuralVergeLinkedInPeopleSearchTool,
    NeuralVergeLinkedInProfileEmailTool,
    NeuralVergeLinkedInProfileFinderTool,
    NeuralVergePersonByEmailTool,
    NeuralVergePhoneLookupTool,
    NeuralVergeResearchTool,
    NeuralVergeUSPhoneLookupTool,
    NeuralVergeWebSearchTool,
)
import pytest
import requests


BASE = "https://api.neuralverge.ai/functions/v1"
ENVELOPE = {
    "session_id": "6ed8818b-c383-475f-ba22-04413d262c35",
    "kind": "email_enrichment",
    "human": "# Email enrichment\n\n- **Full names:** Jane Doe",
    "machine": {"full_names": ["Jane Doe"], "company": "Example Inc."},
    "total_points": 10,
}


@pytest.fixture(autouse=True)
def mock_api_key():
    with patch.dict(os.environ, {"NEURALVERGE_API_KEY": "test_key"}):
        yield


def _response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


CASES = [
    (NeuralVergePersonByEmailTool, {"email": "jane.doe@example.com"},
     "run-email-enrichment", {"email": "jane.doe@example.com"}),
    (NeuralVergeEmailValidationTool, {"email": "jane.doe@example.com"},
     "run-email-validation", {"email": "jane.doe@example.com"}),
    (NeuralVergeEmailFinderTool, {"first_name": "Jane", "last_name": "Doe", "domain": "example.com"},
     "run-email-finder", {"first_name": "Jane", "last_name": "Doe", "domain": "example.com"}),
    (NeuralVergePhoneLookupTool, {"phone": "+15555550100"}, "run-phone-enrichment",
     {"phone": "+15555550100"}),
    (NeuralVergeUSPhoneLookupTool, {"phone": "15555550100"}, "run-phone-enrichment-us",
     {"phone": "15555550100"}),
    (NeuralVergeCompanyFundingTool, {"crunchbase_url": "https://www.crunchbase.com/organization/example"},
     "run-crunchbase-company", {"url": "https://www.crunchbase.com/organization/example"}),
    (NeuralVergeWebSearchTool, {"query": "example query", "max_results": 3}, "run-search",
     {"query": "example query", "settings": {"country": "us", "language": "en", "max_results": 3}}),
    (NeuralVergeExtractTool, {"url": "https://example.com", "instructions": "Get the name"}, "run-extract",
     {"url": "https://example.com", "instructions": "Get the name", "settings": {"country_code": "us"}}),
    (NeuralVergeLinkedInProfileEmailTool, {"profile_url": "https://www.linkedin.com/in/janedoe/"},
     "run-linkedin-email", {"username": "https://www.linkedin.com/in/janedoe/", "includeEmail": True}),
    (NeuralVergeLinkedInProfileFinderTool, {"full_name": "Jane Doe", "company_or_domain": "example.com"},
     "run-linkedin-domain", {"full_name": "Jane Doe", "company_or_domain": "example.com"}),
    (NeuralVergeLinkedInCompanySearchTool, {"search_query": "AI companies", "company_size": ["51-200"]},
     "run-linkedin-company-search", {"searchQuery": "AI companies", "companySize": ["51-200"], "maxItems": 10}),
    (NeuralVergeLinkedInPeopleSearchTool, {"search_query": "recruiter", "current_company": ["Example Inc."]},
     "run-linkedin-people-search", {"searchQuery": "recruiter", "currentCompany": ["Example Inc."], "maxResults": 25}),
    (NeuralVergeLinkedInCompanyEmployeesTool, {"companies": ["https://www.linkedin.com/company/example/"]},
     "run-linkedin-company-employee", {"companies": ["https://www.linkedin.com/company/example/"], "maxResults": 10}),
    (NeuralVergeAmazonProductSearchTool, {"query": "wireless mouse", "max_items": 2},
     "run-amazon-product-search", {"query": "wireless mouse", "domain": "amazon.com", "max_items": 2}),
    (NeuralVergeAmazonProductTool, {"asin": "B004YAVF8I", "domain": "amazon.de"},
     "run-amazon-product", {"asin": "B004YAVF8I", "domain": "amazon.de"}),
    (NeuralVergeAmazonOfferTool, {"asin": "B004YAVF8I"}, "run-amazon-product-offers",
     {"asin": "B004YAVF8I", "domain": "amazon.com"}),
    (NeuralVergeAmazonSellerTool, {"seller": "A2L77EE7U53NWQ"}, "run-amazon-seller",
     {"seller": "A2L77EE7U53NWQ", "domain": "amazon.com"}),
    (NeuralVergeAmazonSellerProductsTool, {"seller": "A2L77EE7U53NWQ", "start_page": 2},
     "run-amazon-seller-products", {"seller": "A2L77EE7U53NWQ", "domain": "amazon.com", "max_items": 20, "start_page": 2}),
]


@pytest.mark.parametrize(("tool_cls", "args", "endpoint", "expected_body"), CASES)
@patch("requests.request")
def test_tool_calls_endpoint(mock_request, tool_cls, args, endpoint, expected_body):
    mock_request.return_value = _response(ENVELOPE)
    tool = tool_cls()
    result = tool.run(**args)

    assert json.loads(result) == ENVELOPE
    method, url = mock_request.call_args.args
    kwargs = mock_request.call_args.kwargs
    assert method == "POST"
    assert url == f"{BASE}/{endpoint}"
    assert kwargs["json"] == expected_body
    assert kwargs["headers"]["x-api-key"] == "test_key"
    assert "Authorization" not in kwargs["headers"]


def test_all_tools_declare_env_var_and_schema():
    for tool_cls, *_ in CASES + [(NeuralVergeResearchTool,)]:
        tool = tool_cls()
        assert tool.name.startswith("NeuralVerge")
        assert tool.args_schema is not None
        assert [e.name for e in tool.env_vars] == ["NEURALVERGE_API_KEY"]


def test_api_key_argument_overrides_env():
    assert NeuralVergeEmailValidationTool(api_key="explicit").api_key == "explicit"


@patch("requests.request")
def test_missing_api_key_returns_failure(mock_request):
    with patch.dict(os.environ, {}, clear=True):
        tool = NeuralVergeEmailValidationTool()
        result = tool._run(email="jane.doe@example.com")
    assert isinstance(result, ToolFailure)
    assert result.code == "missing_api_key"
    mock_request.assert_not_called()


@pytest.mark.parametrize("status", [401, 402, 502])
@patch("requests.request")
def test_http_error_returns_tool_failure(mock_request, status):
    mock_request.return_value = _response({"error": "denied"}, status=status)
    result = NeuralVergeEmailValidationTool()._run(email="jane.doe@example.com")
    assert isinstance(result, ToolFailure)
    assert result.code == str(status)
    assert "denied" in result.message
    assert result.retryable is (status >= 500)


@patch("requests.request", side_effect=requests.ConnectionError("boom"))
def test_network_error_returns_tool_failure(_mock_request):
    result = NeuralVergeEmailValidationTool()._run(email="jane.doe@example.com")
    assert isinstance(result, ToolFailure)
    assert result.code == "network_error"


@patch("time.sleep")
@patch("requests.request")
def test_research_polls_until_complete(mock_request, _sleep):
    sid = "11111111-1111-1111-1111-111111111111"
    mock_request.side_effect = [
        _response({"session_id": sid}),
        _response({"session_id": sid, "status": "running"}),
        _response({
            "session_id": sid,
            "status": "complete",
            "results": {"kind": "deepsearch", "human": "# Report", "machine": {"risk": "low"}, "total_points": 50},
        }),
    ]
    result = json.loads(NeuralVergeResearchTool().run(instructions="Analyze Example Inc."))
    assert result["machine"] == {"risk": "low"}
    assert result["session_id"] == sid

    start, poll, _ = mock_request.call_args_list
    assert start.args == ("POST", f"{BASE}/run-research")
    assert start.kwargs["json"] == {
        "instructions": "Analyze Example Inc.",
        "settings": {"country_code": "us", "search_enabled": True, "deepsearch_model": "base"},
    }
    assert poll.args == ("GET", f"{BASE}/get-session-status")
    assert poll.kwargs["params"] == {"session_id": sid}


@patch("time.sleep")
@patch("requests.request")
def test_research_failed(mock_request, _sleep):
    mock_request.side_effect = [
        _response({"session_id": "s1"}),
        _response({"session_id": "s1", "status": "failed"}),
    ]
    result = NeuralVergeResearchTool()._run(instructions="x")
    assert isinstance(result, ToolFailure)
    assert result.code == "research_failed"


@patch("requests.request")
def test_research_timeout(mock_request):
    mock_request.side_effect = [
        _response({"session_id": "s1"}),
        _response({"session_id": "s1", "status": "queued"}),
    ]
    result = NeuralVergeResearchTool(max_wait=0)._run(instructions="x")
    assert isinstance(result, ToolFailure)
    assert result.code == "timeout"


@pytest.mark.asyncio
@patch("requests.request")
async def test_async_run(mock_request):
    mock_request.return_value = _response(ENVELOPE)
    result = await NeuralVergePersonByEmailTool().arun(email="jane.doe@example.com")
    assert json.loads(result) == ENVELOPE
