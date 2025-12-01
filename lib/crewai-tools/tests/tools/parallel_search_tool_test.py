import json
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.parallel_tools.parallel_search_tool import (
    PARALLEL_AVAILABLE,
    ParallelSearchTool,
)


@pytest.fixture
def mock_parallel_client():
    """Create a mock Parallel client with search response."""
    mock_client = MagicMock()
    mock_client.beta.search.return_value = MagicMock(
        model_dump=MagicMock(
            return_value={
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
        )
    )
    return mock_client


def test_requires_parallel_web_package(monkeypatch):
    """Test that the tool requires parallel-web package to be installed."""
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    with patch(
        "crewai_tools.tools.parallel_tools.parallel_search_tool.PARALLEL_AVAILABLE",
        False,
    ):
        with pytest.raises(ImportError, match="parallel-web"):
            ParallelSearchTool()


def test_requires_env_var(monkeypatch, mock_parallel_client):
    """Test that the tool requires PARALLEL_API_KEY to be set."""
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    with patch(
        "crewai_tools.tools.parallel_tools.parallel_search_tool.PARALLEL_AVAILABLE",
        True,
    ), patch(
        "crewai_tools.tools.parallel_tools.parallel_search_tool.Parallel",
        return_value=mock_parallel_client,
    ):
        with pytest.raises(ValueError, match="PARALLEL_API_KEY"):
            ParallelSearchTool()


@pytest.mark.skipif(not PARALLEL_AVAILABLE, reason="parallel-web not installed")
def test_happy_path(monkeypatch, mock_parallel_client):
    """Test successful search with the Parallel SDK."""
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")

    with patch(
        "crewai_tools.tools.parallel_tools.parallel_search_tool.Parallel",
        return_value=mock_parallel_client,
    ):
        tool = ParallelSearchTool()
        result = tool.run(
            objective="When was the UN established?",
            search_queries=["Founding year UN"],
        )

    data = json.loads(result)
    assert "search_id" in data
    assert "results" in data
    assert len(data["results"]) > 0
    assert data["results"][0]["url"] == "https://www.un.org/en/about-us/history-of-the-un"


@pytest.mark.skipif(not PARALLEL_AVAILABLE, reason="parallel-web not installed")
def test_requires_objective_or_queries(monkeypatch, mock_parallel_client):
    """Test that at least one of objective or search_queries is required."""
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")

    with patch(
        "crewai_tools.tools.parallel_tools.parallel_search_tool.Parallel",
        return_value=mock_parallel_client,
    ):
        tool = ParallelSearchTool()
        result = tool.run()

    assert "Error" in result
    assert "objective" in result.lower() or "search_queries" in result.lower()
