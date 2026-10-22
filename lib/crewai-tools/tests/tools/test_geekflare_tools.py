import json
import os
from unittest.mock import MagicMock, patch

from crewai_tools import (
    GeekflareBrandTool,
    GeekflareScrapeWebsiteTool,
    GeekflareScreenshotTool,
    GeekflareSearchTool,
    GeekflareWebsiteDiagnosticsTool,
)
import pytest


@pytest.fixture(autouse=True)
def mock_geekflare_api_key():
    with patch.dict(os.environ, {"GEEKFLARE_API_KEY": "test_key"}):
        yield


def _response(status: int = 200, body: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = (
        body if body is not None else {"apiStatus": "success", "data": {"ok": True}}
    )
    return response


@patch("requests.post")
def test_search_sends_query_and_config(mock_post):
    mock_post.return_value = _response()
    tool = GeekflareSearchTool(limit=5, serp=True, city="London,England,United Kingdom")

    result = json.loads(tool.run(query="best crm"))

    assert result["data"] == {"ok": True}
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.geekflare.com/search"
    assert kwargs["headers"]["x-api-key"] == "test_key"
    assert kwargs["json"]["query"] == "best crm"
    assert kwargs["json"]["limit"] == 5
    assert kwargs["json"]["serp"] is True
    assert kwargs["json"]["city"] == "London,England,United Kingdom"
    assert "includeDomains" not in kwargs["json"]


@patch("requests.post")
def test_scrape_formats_and_template(mock_post):
    mock_post.return_value = _response()
    tool = GeekflareScrapeWebsiteTool(formats=["markdown-llm"], template="product")

    tool.run(url="https://example.com")

    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.geekflare.com/webscraping"
    assert kwargs["json"]["format"] == ["markdown-llm"]
    assert kwargs["json"]["template"] == "product"
    assert kwargs["json"]["extractionMode"] == "template"


@patch("requests.post")
def test_screenshot_payload(mock_post):
    mock_post.return_value = _response()
    GeekflareScreenshotTool(full_page=True, image_type="webp").run(url="https://example.com")

    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.geekflare.com/screenshot"
    assert kwargs["json"] == {
        "url": "https://example.com",
        "fullPage": True,
        "device": "desktop",
        "type": "webp",
    }


@patch("requests.post")
def test_brand_payload(mock_post):
    mock_post.return_value = _response()
    GeekflareBrandTool(mode="enriched").run(url="https://stripe.com")

    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.geekflare.com/brand"
    assert kwargs["json"] == {"url": "https://stripe.com", "mode": "enriched"}


@pytest.mark.parametrize(
    ("check", "path", "follow_redirect"),
    [
        ("site_status", "up", True),
        ("dns_records", "dnsrecord", False),
        ("dnssec", "dnssec", False),
        ("tls_scan", "tlsscan", False),
        ("redirect_check", "redirectcheck", False),
        ("broken_links", "brokenlink", True),
        ("load_time", "loadtime", True),
        ("mixed_content", "mixedcontent", True),
        ("meta_tags", "metascraping", False),
        ("lighthouse", "lighthouse", True),
    ],
)
@patch("requests.post")
def test_diagnostics_routes_each_check(mock_post, check, path, follow_redirect):
    mock_post.return_value = _response()
    GeekflareWebsiteDiagnosticsTool().run(url="https://example.com", check=check)

    args, kwargs = mock_post.call_args
    assert args[0] == f"https://api.geekflare.com/{path}"
    assert kwargs["json"]["url"] == "https://example.com"
    assert ("followRedirect" in kwargs["json"]) is follow_redirect


@patch("requests.post")
def test_api_error_raises(mock_post):
    mock_post.return_value = _response(401, {"apiCode": 401, "message": "Invalid x-api-key"})

    with pytest.raises(RuntimeError, match="Invalid x-api-key"):
        GeekflareSearchTool()._run(query="x")


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        tool = GeekflareSearchTool()
    with pytest.raises(ValueError, match="GEEKFLARE_API_KEY"):
        tool._run(query="x")


@pytest.mark.asyncio
@patch("requests.post")
async def test_async_run(mock_post):
    mock_post.return_value = _response()
    result = await GeekflareSearchTool()._arun(query="x")
    assert json.loads(result)["data"] == {"ok": True}


def test_env_vars_declared():
    assert [v.name for v in GeekflareSearchTool().env_vars] == ["GEEKFLARE_API_KEY"]
