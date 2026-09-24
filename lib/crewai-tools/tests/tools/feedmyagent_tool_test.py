"""Unit tests for the FeedMyAgent crewAI tools.

No real network calls: ``feedmyagent.FeedMyAgent`` is mocked at the class
level, the same way ``tests/tools/brave_search_tool_test.py`` mocks its
HTTP-backed tool.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from feedmyagent import FeedMyAgentError, Item
import pytest

from crewai_tools.tools.feedmyagent_tool.feedmyagent_tool import (
    FeedMyAgentLatestTool,
    FeedMyAgentReportTool,
    FeedMyAgentSearchTool,
)


def _item(**overrides: object) -> Item:
    defaults = dict(
        id="1",
        title="New prompt-injection technique in MCP tool descriptions",
        summary="Observed a tool description embedding an exfiltration instruction.",
        url="https://example.com/items/1",
        tags=["mcp", "security"],
        score=8.5,
    )
    defaults.update(overrides)
    return Item(**defaults)


PATCH_TARGET = "crewai_tools.tools.feedmyagent_tool.feedmyagent_tool.FeedMyAgent"


# ---------------------------------------------------------------------------
# FeedMyAgentLatestTool
# ---------------------------------------------------------------------------


@patch(PATCH_TARGET)
def test_latest_tool_formats_items(mock_fma_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.latest.return_value = [_item()]
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentLatestTool()
    result = tool.run(tags=["mcp"], use_case="security", limit=5)

    mock_fma_cls.assert_called_once_with(
        base_url="https://api.feedmyagent.com", user_agent="feedmyagent-crewai/0.1"
    )
    mock_client.latest.assert_called_once_with(tags=["mcp"], use_case="security", limit=5)
    assert "New prompt-injection technique in MCP tool descriptions" in result
    assert "[mcp, security]" in result
    assert "https://example.com/items/1" in result
    assert "score=8.5" in result


@patch(PATCH_TARGET)
def test_latest_tool_no_results(mock_fma_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.latest.return_value = []
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentLatestTool()
    result = tool.run()

    assert result == "No matching items found on FeedMyAgent."


@patch(PATCH_TARGET)
def test_latest_tool_formats_api_error(mock_fma_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.latest.side_effect = FeedMyAgentError("invalid_request", "bad tag", 400)
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentLatestTool()
    result = tool.run()

    assert result == "FeedMyAgent error (invalid_request): bad tag"


# ---------------------------------------------------------------------------
# FeedMyAgentSearchTool
# ---------------------------------------------------------------------------


@patch(PATCH_TARGET)
def test_search_tool_passes_query_through(mock_fma_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.query.return_value = [_item(id="2", title="Another item", score=3.0)]
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentSearchTool()
    result = tool.run(text="prompt injection in MCP servers", tags=["mcp"], limit=3)

    mock_client.query.assert_called_once_with(
        "prompt injection in MCP servers", tags=["mcp"], limit=3
    )
    assert "Another item" in result
    assert "score=3.0" in result


def test_search_tool_args_schema_requires_text() -> None:
    from crewai_tools.tools.feedmyagent_tool.feedmyagent_tool import (
        FeedMyAgentSearchInput,
    )

    with pytest.raises(Exception):
        FeedMyAgentSearchInput()  # type: ignore[call-arg]  # missing required `text`


# ---------------------------------------------------------------------------
# FeedMyAgentReportTool
# ---------------------------------------------------------------------------


@patch(PATCH_TARGET)
def test_report_tool_posts_and_formats_confirmation(mock_fma_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.report.return_value = _item(id="pending-1", title="Reported title")
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentReportTool(api_key="ask_test123")
    result = tool.run(
        title="Reported title", description="Full details...", url="https://x.example/1"
    )

    mock_fma_cls.assert_called_once_with(
        api_key="ask_test123",
        base_url="https://api.feedmyagent.com",
        user_agent="feedmyagent-crewai/0.1",
    )
    mock_client.report.assert_called_once_with(
        title="Reported title", description="Full details...", url="https://x.example/1"
    )
    assert result == "Reported to FeedMyAgent: Reported title (https://example.com/items/1)"


@patch(PATCH_TARGET)
def test_report_tool_surfaces_missing_api_key_error(mock_fma_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.report.side_effect = FeedMyAgentError(
        "unauthorized", "An API key is required for this operation.", 401
    )
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentReportTool()  # no api_key set
    result = tool.run(title="t", description="d")

    assert result.startswith("FeedMyAgent error (unauthorized):")
