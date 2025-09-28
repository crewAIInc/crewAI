import json
from unittest.mock import patch
from urllib.parse import urlparse

from crewai_tools.tools.parallel_tools.parallel_search_tool import (
    ParallelSearchTool,
)


def test_requires_env_var(monkeypatch):
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    tool = ParallelSearchTool()
    result = tool.run(objective="test")
    assert "PARALLEL_API_KEY" in result


@patch("crewai_tools.tools.parallel_tools.parallel_search_tool.requests.post")
def test_happy_path(mock_post, monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "test")

    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "search_id": "search_123",
        "results": [
            {
                "url": "https://www.un.org/en/about-us/history-of-the-un",
                "title": "History of the United Nations",
                "excerpts": [
                    "Four months after the San Francisco Conference ended, the United Nations officially began, on 24 October 1945..."
                ],
            }
        ],
    }

    tool = ParallelSearchTool()
    result = tool.run(
        objective="When was the UN established?", search_queries=["Founding year UN"]
    )
    data = json.loads(result)
    assert "search_id" in data
    urls = [r.get("url", "") for r in data.get("results", [])]
    # Validate host against allowed set instead of substring matching
    allowed_hosts = {"www.un.org", "un.org"}
    assert any(urlparse(u).netloc in allowed_hosts for u in urls)
