from typing import Any

from crewai_tools.tools.tavily_research_tool import tavily_research_tool
from crewai_tools.tools.tavily_research_tool.tavily_research_tool import (
    TavilyResearchTool,
)


class _FakeTavilyClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    def research(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


def test_tavily_research_tool_accepts_output_schema_default(monkeypatch):
    output_schema = {
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    monkeypatch.setattr(tavily_research_tool, "TAVILY_AVAILABLE", True)
    monkeypatch.setattr(
        tavily_research_tool, "TavilyClient", _FakeTavilyClient, raising=False
    )
    monkeypatch.setattr(
        tavily_research_tool, "AsyncTavilyClient", _FakeTavilyClient, raising=False
    )

    tool = TavilyResearchTool(output_schema=output_schema)

    assert tool.output_schema == output_schema


def test_tavily_research_tool_drops_transitional_schema_keyword(monkeypatch):
    output_schema = {
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }
    tavily_output_schema = {
        "properties": {"legacy_summary": {"type": "string"}},
        "required": ["legacy_summary"],
    }

    monkeypatch.setattr(tavily_research_tool, "TAVILY_AVAILABLE", True)
    monkeypatch.setattr(
        tavily_research_tool, "TavilyClient", _FakeTavilyClient, raising=False
    )
    monkeypatch.setattr(
        tavily_research_tool, "AsyncTavilyClient", _FakeTavilyClient, raising=False
    )

    tool = TavilyResearchTool(
        output_schema=output_schema,
        tavily_output_schema=tavily_output_schema,
    )

    assert tool.output_schema == output_schema


def test_tavily_research_tool_maps_transitional_schema_keyword(monkeypatch):
    tavily_output_schema = {
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    monkeypatch.setattr(tavily_research_tool, "TAVILY_AVAILABLE", True)
    monkeypatch.setattr(
        tavily_research_tool, "TavilyClient", _FakeTavilyClient, raising=False
    )
    monkeypatch.setattr(
        tavily_research_tool, "AsyncTavilyClient", _FakeTavilyClient, raising=False
    )

    tool = TavilyResearchTool(tavily_output_schema=tavily_output_schema)

    assert tool.output_schema == tavily_output_schema
