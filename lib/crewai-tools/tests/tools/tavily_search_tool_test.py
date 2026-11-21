import os
from unittest.mock import patch

from crewai_tools import TavilyExtractorTool, TavilySearchTool


def test_tavily_search_tool_uses_the_given_api_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "crewai_tools.tools.tavily_search_tool.tavily_search_tool.TavilyClient"
        ) as mock_client, patch(
            "crewai_tools.tools.tavily_search_tool.tavily_search_tool.AsyncTavilyClient"
        ) as mock_async_client:
            tool = TavilySearchTool(
                api_key="test_api_key",
                api_base_url="https://custom.tavily.api.com",
            )

            assert tool.api_base_url == "https://custom.tavily.api.com"
            expected = {
                "api_key": "test_api_key",
                "proxies": None,
                "api_base_url": "https://custom.tavily.api.com",
                "client_name": "crewai",
            }
            mock_client.assert_called_once_with(**expected)
            mock_async_client.assert_called_once_with(**expected)


def test_tavily_search_tool_reads_api_base_url_from_the_environment():
    with patch.dict(
        os.environ,
        {
            "TAVILY_API_KEY": "env_api_key",
            "TAVILY_API_BASE_URL": "https://env.tavily.api.com",
        },
        clear=True,
    ):
        with patch(
            "crewai_tools.tools.tavily_search_tool.tavily_search_tool.TavilyClient"
        ) as mock_client, patch(
            "crewai_tools.tools.tavily_search_tool.tavily_search_tool.AsyncTavilyClient"
        ) as mock_async_client:
            tool = TavilySearchTool()

            assert tool.api_base_url == "https://env.tavily.api.com"
            expected = {
                "api_key": "env_api_key",
                "proxies": None,
                "api_base_url": "https://env.tavily.api.com",
                "client_name": "crewai",
            }
            mock_client.assert_called_once_with(**expected)
            mock_async_client.assert_called_once_with(**expected)


def test_tavily_search_tool_without_api_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "crewai_tools.tools.tavily_search_tool.tavily_search_tool.TavilyClient"
        ) as mock_client, patch(
            "crewai_tools.tools.tavily_search_tool.tavily_search_tool.AsyncTavilyClient"
        ) as mock_async_client:
            tool = TavilySearchTool(api_key="test_api_key")

            assert tool.api_base_url is None
            expected = {
                "api_key": "test_api_key",
                "proxies": None,
                "api_base_url": None,
                "client_name": "crewai",
            }
            mock_client.assert_called_once_with(**expected)
            mock_async_client.assert_called_once_with(**expected)


def test_tavily_extractor_tool_uses_the_given_api_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "crewai_tools.tools.tavily_extractor_tool.tavily_extractor_tool.TavilyClient"
        ) as mock_client, patch(
            "crewai_tools.tools.tavily_extractor_tool.tavily_extractor_tool.AsyncTavilyClient"
        ) as mock_async_client:
            tool = TavilyExtractorTool(
                api_key="test_api_key",
                api_base_url="https://custom.tavily.api.com",
            )

            assert tool.api_base_url == "https://custom.tavily.api.com"
            expected = {
                "api_key": "test_api_key",
                "proxies": None,
                "api_base_url": "https://custom.tavily.api.com",
                "client_name": "crewai",
            }
            mock_client.assert_called_once_with(**expected)
            mock_async_client.assert_called_once_with(**expected)


def test_tavily_extractor_tool_reads_api_base_url_from_the_environment():
    with patch.dict(
        os.environ,
        {
            "TAVILY_API_KEY": "env_api_key",
            "TAVILY_API_BASE_URL": "https://env.tavily.api.com",
        },
        clear=True,
    ):
        with patch(
            "crewai_tools.tools.tavily_extractor_tool.tavily_extractor_tool.TavilyClient"
        ) as mock_client, patch(
            "crewai_tools.tools.tavily_extractor_tool.tavily_extractor_tool.AsyncTavilyClient"
        ) as mock_async_client:
            tool = TavilyExtractorTool()

            assert tool.api_base_url == "https://env.tavily.api.com"
            expected = {
                "api_key": "env_api_key",
                "proxies": None,
                "api_base_url": "https://env.tavily.api.com",
                "client_name": "crewai",
            }
            mock_client.assert_called_once_with(**expected)
            mock_async_client.assert_called_once_with(**expected)
