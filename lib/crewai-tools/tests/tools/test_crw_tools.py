from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from crewai_tools.tools.crw_tool.crw_scrape_tool import CrwScrapeWebsiteTool
from crewai_tools.tools.crw_tool.crw_map_tool import CrwMapWebsiteTool
from crewai_tools.tools.crw_tool.crw_crawl_tool import CrwCrawlWebsiteTool


# --- CrwScrapeWebsiteTool ---


@patch("crewai_tools.tools.crw_tool.crw_scrape_tool.http_requests")
def test_scrape_returns_markdown(mock_requests):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "success": True,
        "data": {"markdown": "# Hello"},
    }
    mock_resp.raise_for_status = MagicMock()
    mock_requests.post.return_value = mock_resp

    tool = CrwScrapeWebsiteTool(api_url="http://localhost:3000")
    result = tool._run(url="https://example.com")

    assert result == "# Hello"
    mock_requests.post.assert_called_once()


@patch("crewai_tools.tools.crw_tool.crw_scrape_tool.http_requests")
def test_scrape_raises_on_failure(mock_requests):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"success": False, "error": "bad url"}
    mock_resp.raise_for_status = MagicMock()
    mock_requests.post.return_value = mock_resp

    tool = CrwScrapeWebsiteTool(api_url="http://localhost:3000")
    with pytest.raises(RuntimeError, match="CRW scrape failed"):
        tool._run(url="https://bad.example.com")


# --- CrwMapWebsiteTool ---


@patch("crewai_tools.tools.crw_tool.crw_map_tool.http_requests")
def test_map_returns_links(mock_requests):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "success": True,
        "data": {"links": ["https://a.com", "https://b.com"]},
    }
    mock_resp.raise_for_status = MagicMock()
    mock_requests.post.return_value = mock_resp

    tool = CrwMapWebsiteTool(api_url="http://localhost:3000")
    result = tool._run(url="https://example.com")

    assert "https://a.com" in result
    assert "https://b.com" in result


@patch("crewai_tools.tools.crw_tool.crw_map_tool.http_requests")
def test_map_raises_on_failure(mock_requests):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"success": False, "error": "not found"}
    mock_resp.raise_for_status = MagicMock()
    mock_requests.post.return_value = mock_resp

    tool = CrwMapWebsiteTool(api_url="http://localhost:3000")
    with pytest.raises(RuntimeError, match="CRW map failed"):
        tool._run(url="https://example.com")


# --- CrwCrawlWebsiteTool ---


@patch("crewai_tools.tools.crw_tool.crw_crawl_tool.time")
@patch("crewai_tools.tools.crw_tool.crw_crawl_tool.http_requests")
def test_crawl_returns_combined_markdown(mock_requests, mock_time):
    mock_time.sleep = MagicMock()

    start_resp = MagicMock()
    start_resp.json.return_value = {"success": True, "id": "job-1"}
    start_resp.raise_for_status = MagicMock()

    status_resp = MagicMock()
    status_resp.json.return_value = {
        "status": "completed",
        "data": [
            {"metadata": {"sourceURL": "https://example.com"}, "markdown": "# Page 1"},
            {"metadata": {"sourceURL": "https://example.com/about"}, "markdown": "# About"},
        ],
    }
    status_resp.raise_for_status = MagicMock()

    mock_requests.post.return_value = start_resp
    mock_requests.get.return_value = status_resp

    tool = CrwCrawlWebsiteTool(api_url="http://localhost:3000")
    result = tool._run(url="https://example.com")

    assert "# Page 1" in result
    assert "# About" in result


@patch("crewai_tools.tools.crw_tool.crw_crawl_tool.time")
@patch("crewai_tools.tools.crw_tool.crw_crawl_tool.http_requests")
def test_crawl_failed_includes_error_detail(mock_requests, mock_time):
    mock_time.sleep = MagicMock()

    start_resp = MagicMock()
    start_resp.json.return_value = {"success": True, "id": "job-2"}
    start_resp.raise_for_status = MagicMock()

    status_resp = MagicMock()
    status_resp.json.return_value = {
        "status": "failed",
        "error": "Rate limit exceeded",
    }
    status_resp.raise_for_status = MagicMock()

    mock_requests.post.return_value = start_resp
    mock_requests.get.return_value = status_resp

    tool = CrwCrawlWebsiteTool(api_url="http://localhost:3000")
    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        tool._run(url="https://example.com")


def test_crawl_poll_interval_must_be_positive():
    with pytest.raises(ValidationError):
        CrwCrawlWebsiteTool(api_url="http://localhost:3000", poll_interval=0)


def test_crawl_max_wait_must_be_positive():
    with pytest.raises(ValidationError):
        CrwCrawlWebsiteTool(api_url="http://localhost:3000", max_wait=-1)
