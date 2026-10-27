import os
from unittest.mock import patch

from crewai_tools import TavilyExtractorTool, TavilySearchTool
import pytest


TOOLS = [
    (TavilySearchTool, "crewai_tools.tools.tavily_search_tool.tavily_search_tool"),
    (TavilyExtractorTool, "crewai_tools.tools.tavily_extractor_tool.tavily_extractor_tool"),
]


@pytest.mark.parametrize(("tool_cls", "module"), TOOLS)
def test_clients_get_explicit_api_base_url(tool_cls, module):
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(f"{module}.TAVILY_AVAILABLE", True),
        patch(f"{module}.TavilyClient") as sync_client,
        patch(f"{module}.AsyncTavilyClient") as async_client,
    ):
        tool = tool_cls(api_key="test_key", api_base_url="https://tavily.example.com")

        assert tool.api_base_url == "https://tavily.example.com"
        for client in (sync_client, async_client):
            client.assert_called_once_with(
                api_key="test_key",
                proxies=None,
                api_base_url="https://tavily.example.com",
                client_name="crewai",
            )


@pytest.mark.parametrize(("tool_cls", "module"), TOOLS)
def test_api_base_url_defaults_to_env(tool_cls, module):
    with (
        patch.dict(os.environ, {"TAVILY_API_BASE_URL": "https://env.tavily.example.com"}, clear=True),
        patch(f"{module}.TAVILY_AVAILABLE", True),
        patch(f"{module}.TavilyClient") as sync_client,
        patch(f"{module}.AsyncTavilyClient") as async_client,
    ):
        tool = tool_cls(api_key="test_key")

        assert tool.api_base_url == "https://env.tavily.example.com"
        assert sync_client.call_args.kwargs["api_base_url"] == "https://env.tavily.example.com"
        assert async_client.call_args.kwargs["api_base_url"] == "https://env.tavily.example.com"


@pytest.mark.parametrize(("tool_cls", "module"), TOOLS)
def test_api_base_url_unset_keeps_tavily_default(tool_cls, module):
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(f"{module}.TAVILY_AVAILABLE", True),
        patch(f"{module}.TavilyClient") as sync_client,
        patch(f"{module}.AsyncTavilyClient"),
    ):
        tool = tool_cls(api_key="test_key")

        assert tool.api_base_url is None
        assert sync_client.call_args.kwargs["api_base_url"] is None
