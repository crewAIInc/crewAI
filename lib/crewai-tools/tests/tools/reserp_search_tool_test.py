"""Contract tests for ReserpSearchTool."""

from unittest.mock import patch

import pytest
from crewai_tools.tools.reserp_search_tool.reserp_search_tool import ReserpSearchTool


@pytest.fixture(autouse=True)
def reserp_api_key(monkeypatch):
    monkeypatch.setenv("RESERP_API_KEY", "test-key")


def test_public_export():
    from crewai_tools import ReserpSearchTool as PublicReserpSearchTool

    assert PublicReserpSearchTool is ReserpSearchTool


@patch("crewai_tools.tools.reserp_search_tool.reserp_search_tool.requests.post")
def test_one_request_and_unchanged_payload(post):
    payload = {
        "ok": True,
        "url": "https://www.google.com/search?q=test",
        "finalUrl": "https://www.google.com/search?q=test",
        "results": [{"url": "https://example.com"}],
        "pagination": {
            "start": 0,
            "nextStart": 10,
            "nextUrl": "https://www.google.com/search?q=test&start=10",
        },
        "billed": True,
    }
    post.return_value.json.return_value = payload

    result = ReserpSearchTool().run(
        url="https://www.google.com/search?q=test"
    )

    assert result == payload
    post.assert_called_once_with(
        "https://api.reserp.ai/v1/serp",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        json={"url": "https://www.google.com/search?q=test"},
    )
    post.return_value.raise_for_status.assert_called_once_with()


@patch("crewai_tools.tools.reserp_search_tool.reserp_search_tool.requests.post")
def test_failure_propagates_without_retry(post):
    failure = RuntimeError("transport failure")
    post.side_effect = failure

    with pytest.raises(RuntimeError) as caught:
        ReserpSearchTool().run(url="https://www.google.com/search?q=test")

    assert caught.value is failure
    post.assert_called_once()
