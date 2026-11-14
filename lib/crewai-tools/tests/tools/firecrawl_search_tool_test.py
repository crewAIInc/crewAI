import pytest

from crewai_tools.tools.firecrawl_search_tool.firecrawl_search_tool import FirecrawlSearchTool


@pytest.mark.vcr()
def test_firecrawl_search_tool_integration():
    tool = FirecrawlSearchTool()
    result = tool.run(query="firecrawl")

    assert result is not None
    assert hasattr(result, 'web') or hasattr(result, 'news') or hasattr(result, 'images')


from unittest.mock import patch


def test_search_tool_accepts_explicit_none_config():
    """config=None means "no extra Firecrawl options"; it must not raise TypeError."""
    with patch("firecrawl.FirecrawlApp") as app:
        app.return_value.search.return_value = {"data": []}
        tool = FirecrawlSearchTool(api_key="test", config=None)
        result = tool._run(query="crewai")

    app.return_value.search.assert_called_once_with(query="crewai")
    assert result == {"data": []}

