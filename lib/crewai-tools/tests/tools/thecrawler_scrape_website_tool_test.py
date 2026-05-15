import os
from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

from crewai_tools.tools.thecrawler_scrape_website_tool.thecrawler_scrape_website_tool import (
    TheCrawlerScrapeWebsiteTool,
)


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    resp = MagicMock(spec=requests_lib.Response)
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


@pytest.fixture(autouse=True)
def _thecrawler_env():
    with patch.dict(os.environ, {"THECRAWLER_API_KEY": "mai_live_" + "a" * 32}):
        yield


def test_thecrawler_scrape_tool_init_reads_env() -> None:
    tool = TheCrawlerScrapeWebsiteTool()
    assert tool.api_key == "mai_live_" + "a" * 32
    assert tool.api_base == "https://www.miaibot.ai/api/v1"
    assert tool.config == {"extractMarkdown": True}


def test_thecrawler_scrape_tool_init_explicit_key() -> None:
    tool = TheCrawlerScrapeWebsiteTool(api_key="mai_live_" + "b" * 32)
    assert tool.api_key == "mai_live_" + "b" * 32


def test_thecrawler_scrape_tool_run_posts_to_crawl_endpoint() -> None:
    tool = TheCrawlerScrapeWebsiteTool()
    mock_resp = _mock_response(
        status_code=200,
        json_data={
            "ok": True,
            "items": [
                {
                    "status": "success",
                    "url": "https://example.com",
                    "markdown": "# Example Domain\n\nExample text.",
                    "title": "Example Domain",
                    "errorType": None,
                    "errorRetryable": False,
                }
            ],
            "successCount": 1,
            "errorCount": 0,
            "creditsCharged": 1,
            "balance": 999,
        },
    )

    with patch(
        "crewai_tools.tools.thecrawler_scrape_website_tool.thecrawler_scrape_website_tool.requests.post",
        return_value=mock_resp,
    ) as mock_post:
        result = tool.run(url="https://example.com")

    mock_post.assert_called_once()
    called_url = mock_post.call_args.args[0]
    called_kwargs = mock_post.call_args.kwargs
    assert called_url == "https://www.miaibot.ai/api/v1/crawl"
    assert called_kwargs["json"] == {
        "urls": ["https://example.com"],
        "extractMarkdown": True,
    }
    assert called_kwargs["headers"]["Authorization"].startswith("Bearer mai_live_")
    assert called_kwargs["headers"]["Content-Type"] == "application/json"

    assert result["ok"] is True
    assert result["items"][0]["markdown"].startswith("# Example Domain")


def test_thecrawler_scrape_tool_custom_config_merged() -> None:
    # The default config has `extractMarkdown: True`. A caller passing a custom
    # config should ADD their fields to the defaults, not REPLACE the whole map.
    tool = TheCrawlerScrapeWebsiteTool(
        config={"usePlaywright": True, "retries": 5},
    )
    mock_resp = _mock_response(
        status_code=200, json_data={"ok": True, "items": []}
    )

    with patch(
        "crewai_tools.tools.thecrawler_scrape_website_tool.thecrawler_scrape_website_tool.requests.post",
        return_value=mock_resp,
    ) as mock_post:
        tool.run(url="https://example.com")

    sent = mock_post.call_args.kwargs["json"]
    assert sent["urls"] == ["https://example.com"]
    # Default field survives the merge.
    assert sent["extractMarkdown"] is True
    # Caller-supplied fields are applied.
    assert sent["usePlaywright"] is True
    assert sent["retries"] == 5


def test_thecrawler_scrape_tool_config_urls_cannot_override_runtime_url() -> None:
    # Security: if a caller (or a downstream config source) sneaks a `urls` key
    # into config, the runtime-validated `url` arg must still win. Otherwise
    # validate_url() would be bypassed.
    tool = TheCrawlerScrapeWebsiteTool(
        config={"urls": ["https://malicious.test"]},
    )
    mock_resp = _mock_response(
        status_code=200, json_data={"ok": True, "items": []}
    )

    with patch(
        "crewai_tools.tools.thecrawler_scrape_website_tool.thecrawler_scrape_website_tool.requests.post",
        return_value=mock_resp,
    ) as mock_post:
        tool.run(url="https://example.com")

    sent = mock_post.call_args.kwargs["json"]
    assert sent["urls"] == ["https://example.com"]


def test_thecrawler_scrape_tool_missing_key_raises() -> None:
    with patch.dict(os.environ, {}, clear=True):
        tool = TheCrawlerScrapeWebsiteTool()
        with pytest.raises(ValueError, match="API key"):
            tool.run(url="https://example.com")


def test_thecrawler_scrape_tool_custom_api_base() -> None:
    tool = TheCrawlerScrapeWebsiteTool(api_base="https://api.example.test/v1")
    mock_resp = _mock_response(
        status_code=200, json_data={"ok": True, "items": []}
    )

    with patch(
        "crewai_tools.tools.thecrawler_scrape_website_tool.thecrawler_scrape_website_tool.requests.post",
        return_value=mock_resp,
    ) as mock_post:
        tool.run(url="https://example.com")

    assert mock_post.call_args.args[0] == "https://api.example.test/v1/crawl"
