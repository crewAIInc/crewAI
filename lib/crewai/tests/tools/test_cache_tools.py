from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.cache_tools.cache_tools import CacheTools
import pytest


@pytest.mark.parametrize(
    "query",
    [
        "ordinary query",
        "documentation for tool: search",
        "tool:search",
        "log entry |input: original query",
        "tool:search |input: original query",
    ],
)
def test_cache_tool_preserves_delimiters_in_input(query: str) -> None:
    """Retrieve the exact cached input through the public structured tool."""
    cache = CacheHandler()
    cache.add("search", query, "matching result")
    cache.add("search", "log entry", "truncated input result")
    reader = CacheTools(cache_handler=cache).tool()

    assert reader.invoke({"key": f"tool: search |input: {query}"}) == "matching result"


def test_cache_tool_preserves_missing_entry_behavior() -> None:
    """A well-formed key without a cached value remains a cache miss."""
    reader = CacheTools().tool()

    assert reader.invoke({"key": "tool: search |input: missing"}) is None
