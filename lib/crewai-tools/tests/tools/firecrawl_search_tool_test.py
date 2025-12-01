import pytest

from crewai_tools.tools.firecrawl_search_tool.firecrawl_search_tool import FirecrawlSearchTool


@pytest.mark.vcr()
def test_firecrawl_search_tool_integration():
    tool = FirecrawlSearchTool()
    result = tool.run(query="firecrawl")

    assert result is not None
    assert hasattr(result, 'web') or hasattr(result, 'news') or hasattr(result, 'images')
