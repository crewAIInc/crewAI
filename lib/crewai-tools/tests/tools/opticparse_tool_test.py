"""Unit tests for OpticParse and PhishVision tools."""

from unittest.mock import Mock, patch
import pytest

from crewai_tools.tools.opticparse_tool import (
    OpticParseTool,
    PhishVisionTool,
)


@pytest.fixture
def opticparse_tool():
    """Create OpticParseTool instance."""
    return OpticParseTool(api_key="op_test_12345")


@pytest.fixture
def phishvision_tool():
    """Create PhishVisionTool instance."""
    return PhishVisionTool(api_key="op_test_12345")


def test_opticparse_tool_initialization(opticparse_tool):
    """Test tool initialization and default properties."""
    assert opticparse_tool.name == "OpticParse Web Scraper"
    assert opticparse_tool.api_key == "op_test_12345"
    assert opticparse_tool.timeout == 35


def test_phishvision_tool_initialization(phishvision_tool):
    """Test tool initialization and default properties."""
    assert phishvision_tool.name == "PhishVision Threat Scanner"
    assert phishvision_tool.api_key == "op_test_12345"
    assert phishvision_tool.timeout == 15


def test_insecure_http_portal_rejected_with_api_key():
    """Ensure HTTP portal is rejected when an API key is configured."""
    with pytest.raises(ValueError, match="Insecure HTTP portal_url is prohibited"):
        OpticParseTool(api_key="test_key", portal_url="http://insecure-gateway.com")

    with pytest.raises(ValueError, match="Insecure HTTP portal_url is prohibited"):
        PhishVisionTool(api_key="test_key", portal_url="http://insecure-gateway.com")


def test_invalid_url_validation(opticparse_tool, phishvision_tool):
    """Test URL validation rejecting invalid schemas."""
    res1 = opticparse_tool.run(website_url="not-a-valid-url", extraction_query="get prices")
    assert "Error: Invalid or malformed URL" in res1

    res2 = phishvision_tool.run(website_url="ftp://malicious.com")
    assert "Error: Invalid or malformed URL" in res2


@patch("requests.post")
def test_phishvision_bare_domain_normalization(mock_post, phishvision_tool):
    """Test that bare domains like example.com are normalized to https://."""
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "url": "https://example.com",
        "verdict": "BENIGN_AUDITED",
        "is_phishing": False,
        "confidence_score": 96
    }
    mock_post.return_value = mock_resp

    result = phishvision_tool.run(website_url="example.com")
    assert "BENIGN_AUDITED" in result
    # Verify post was called with normalized url
    called_payload = mock_post.call_args[1]["json"]
    assert called_payload["url"] == "https://example.com"


@patch("requests.post")
def test_redirect_rejection_for_credential_safety(mock_post, opticparse_tool):
    """Ensure redirects are rejected with an error instead of leaking credentials."""
    mock_resp = Mock()
    mock_resp.status_code = 302
    mock_post.return_value = mock_resp

    res = opticparse_tool.run(website_url="https://example.com", extraction_query="test query")
    assert "Redirects are disabled for security" in res


@patch("requests.post")
def test_opticparse_scrape_success(mock_post, opticparse_tool):
    """Test successful scraping execution."""
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "extracted_data": {"price": "$99", "item": "Cloud Pro Tier"}
    }
    mock_post.return_value = mock_resp

    result = opticparse_tool.run(
        website_url="https://example.com/pricing",
        extraction_query="Extract product pricing"
    )
    assert "Cloud Pro Tier" in result
    assert "$99" in result
