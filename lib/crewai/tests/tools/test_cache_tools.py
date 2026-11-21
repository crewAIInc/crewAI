"""Tests for the Hit Cache tool's key handling."""

from crewai.agents.cache.cache_handler import CacheHandler
from crewai.tools.cache_tools.cache_tools import CacheTools
import pytest


TOOL_INPUT = '{"q": "weather"}'


def _cache_tools() -> CacheTools:
    handler = CacheHandler()
    handler.add(tool="search_web", input=TOOL_INPUT, output="sunny, 22C")
    return CacheTools(cache_handler=handler)


def test_hit_cache_reads_composite_key() -> None:
    cache_tools = _cache_tools()

    assert (
        cache_tools.hit_cache(f"tool: search_web|input: {TOOL_INPUT}") == "sunny, 22C"
    )


def test_hit_cache_returns_none_for_composite_miss() -> None:
    cache_tools = _cache_tools()

    assert cache_tools.hit_cache("tool: search_web|input: never cached") is None


@pytest.mark.parametrize(
    "key",
    [
        f"search_web-{TOOL_INPUT}",
        "search_web",
        "",
        TOOL_INPUT,
        "tool: search_web",
        "tool: |input: x",
    ],
)
def test_hit_cache_treats_unparseable_key_as_miss(key: str) -> None:
    cache_tools = _cache_tools()

    assert cache_tools.hit_cache(key) is None


def test_structured_tool_treats_unparseable_key_as_miss() -> None:
    tool = _cache_tools().tool()

    assert tool.invoke({"key": f"search_web-{TOOL_INPUT}"}) is None
