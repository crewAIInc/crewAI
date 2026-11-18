"""Tests for TavilyResearchTool argument resolution."""

from __future__ import annotations

from typing import Any

from crewai_tools.tools.tavily_research_tool.tavily_research_tool import (
    TavilyResearchTool,
)


class _Recorder:
    """Tavily client stub that records the research request."""

    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def research(self, **kwargs: Any) -> dict[str, Any]:
        self.request = kwargs
        return {"ok": True}


def test_tool_defaults_survive_the_per_call_schema() -> None:
    """Configured defaults must reach the client when the call omits them."""
    tool = TavilyResearchTool.model_construct(
        model="pro", stream=True, citation_format="apa"
    )
    recorder = _Recorder()
    tool._client = recorder

    tool.run(input="compare vector databases")

    assert recorder.request["model"] == "pro"
    assert recorder.request["stream"] is True
    assert recorder.request["citation_format"] == "apa"
    assert recorder.request["output_schema"] is None


def test_per_call_values_still_override_the_configured_defaults() -> None:
    tool = TavilyResearchTool.model_construct(
        model="pro", stream=True, citation_format="apa"
    )
    recorder = _Recorder()
    tool._client = recorder

    tool.run(
        input="compare vector databases",
        model="mini",
        stream=False,
        citation_format="mla",
        output_schema={"type": "object"},
    )

    assert recorder.request["model"] == "mini"
    assert recorder.request["stream"] is False
    assert recorder.request["citation_format"] == "mla"
    assert recorder.request["output_schema"] == {"type": "object"}
