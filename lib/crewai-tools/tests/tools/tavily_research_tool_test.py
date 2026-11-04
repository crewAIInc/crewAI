from unittest.mock import MagicMock, patch

from crewai_tools.tools.tavily_research_tool import tavily_research_tool
from crewai_tools.tools.tavily_research_tool.tavily_research_tool import (
    TavilyResearchTool,
)


def _tool_and_client(**kwargs):
    client = MagicMock()
    client.research.return_value = {"request_id": "r1"}
    with (
        patch.object(tavily_research_tool, "TAVILY_AVAILABLE", True),
        patch.object(
            tavily_research_tool,
            "TavilyClient",
            MagicMock(return_value=client),
            create=True,
        ),
        patch.object(
            tavily_research_tool, "AsyncTavilyClient", MagicMock(), create=True
        ),
    ):
        tool = TavilyResearchTool(**kwargs)
    return tool, client


def test_run_uses_constructor_options():
    tool, client = _tool_and_client(model="pro", citation_format="apa")

    tool.run(input="history of the transistor")

    kwargs = client.research.call_args.kwargs
    assert kwargs["model"] == "pro"
    assert kwargs["citation_format"] == "apa"
    assert kwargs["stream"] is False


def test_run_arguments_override_constructor_options():
    tool, client = _tool_and_client(model="pro", citation_format="apa")

    tool.run(input="history of the transistor", model="mini", citation_format="mla")

    kwargs = client.research.call_args.kwargs
    assert kwargs["model"] == "mini"
    assert kwargs["citation_format"] == "mla"
