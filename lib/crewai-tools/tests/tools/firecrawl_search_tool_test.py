from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.firecrawl_search_tool.firecrawl_search_tool import FirecrawlSearchTool


@pytest.mark.vcr()
def test_firecrawl_search_tool_integration():
    tool = FirecrawlSearchTool()
    result = tool.run(query="firecrawl")

    assert result is not None
    assert hasattr(result, 'web') or hasattr(result, 'news') or hasattr(result, 'images')


def test_search_tool_accepts_explicit_none_config():
    """An explicit config=None means no extra search options, not a crash."""
    with patch("firecrawl.FirecrawlApp"):
        tool = FirecrawlSearchTool(api_key="test-key", config=None)

    tool._firecrawl = MagicMock()

    tool._run(query="crewai")

    tool._firecrawl.search.assert_called_once_with(query="crewai")
