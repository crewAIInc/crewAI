import os
import pytest
from dlg4_grant_system import WebSearchTool

@pytest.mark.parametrize(
    "kwargs, expected_query",
    [
        ({"query": "test query"}, "test query"),
        ({"search_query": "test search query"}, "test search query"),
        ({"description": "test description"}, "test description"),
        ({"messages": [{"role": "user", "content": "test from message"}]}, "test from message"),
        ({"input": "test from raw input"}, "test from raw input"),
    ],
)
def test_web_search_tool_finds_query(kwargs, expected_query):
    """Tests that WebSearchTool can find the query from various argument shapes in smoke mode."""
    tool = WebSearchTool(smoke_mode=True)
    result = tool.run(**kwargs)
    assert result == f"This is a smoke test search result for query: {expected_query}"

def test_web_search_tool_no_query():
    """Tests that WebSearchTool fails gracefully when no query is provided."""
    tool = WebSearchTool(smoke_mode=True)
    result = tool.run()
    assert "Search failed: no query provided" in result

def test_web_search_tool_env_var_smoke_mode(monkeypatch):
    """Tests that DLG4_SMOKE=1 environment variable enables smoke mode."""
    monkeypatch.setenv("DLG4_SMOKE", "1")
    tool = WebSearchTool() # smoke_mode is not passed, should be picked from env
    result = tool.run(query="test")
    assert "smoke test search result" in result

def test_web_search_tool_live_error_wrapping(monkeypatch):
    """Tests that live search errors are wrapped gracefully."""

    class MockSerper:
        def run(self, **kwargs):
            raise ValueError("Serper API error: Quota exceeded")

    monkeypatch.setattr(WebSearchTool, "_serper", MockSerper())
    
    tool = WebSearchTool(smoke_mode=False)
    result = tool.run(query="test")
    assert "Search failed due to an exception" in result
    assert "Serper API error: Quota exceeded" in result
