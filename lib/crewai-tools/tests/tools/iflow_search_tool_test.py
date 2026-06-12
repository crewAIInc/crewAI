import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.iflow_search_tool.base import IFlowSearchToolBase
from crewai_tools.tools.iflow_search_tool.iflow_image_search_tool import (
    IFlowImageSearchTool,
)
from crewai_tools.tools.iflow_search_tool.iflow_web_fetch_tool import (
    IFlowWebFetchTool,
)
from crewai_tools.tools.iflow_search_tool.iflow_web_search_tool import (
    IFlowWebSearchTool,
)


@pytest.fixture(autouse=True)
def _iflow_env():
    """Provide an API key for every test unless a test clears it explicitly."""
    with patch.dict(os.environ, {"IFLOW_API_KEY": "test-api-key"}):
        yield


def _web_search_response() -> SimpleNamespace:
    return SimpleNamespace(
        query="crewai",
        took_ms=42,
        results=[
            SimpleNamespace(
                title="CrewAI docs",
                url="https://docs.crewai.com",
                snippet="Framework for orchestrating agents.",
                position=1,
                date="2026-06-01",
            ),
            SimpleNamespace(
                title="CrewAI repo",
                url="https://github.com/crewAIInc/crewAI",
                snippet="Source code.",
                position=2,
                date=None,
            ),
        ],
    )


def _image_search_response() -> SimpleNamespace:
    return SimpleNamespace(
        query="kittens",
        took_ms=21,
        images=[
            SimpleNamespace(
                image_url="https://img.example.com/1.jpg",
                source_url="https://example.com/page",
                title="A kitten",
                width=800,
                height=600,
                position=1,
            ),
        ],
    )


def _web_fetch_response() -> SimpleNamespace:
    return SimpleNamespace(
        url="https://example.com",
        title="Example Domain",
        content="Example page body.",
        from_cache=False,
        took_ms=11,
    )


class TestIFlowWebSearchTool:
    def test_run_returns_normalized_results_json(self):
        client = MagicMock()
        client.web_search.return_value = _web_search_response()
        tool = IFlowWebSearchTool(client=client)

        output = tool.run(query="crewai")
        data = json.loads(output)

        client.web_search.assert_called_once_with(query="crewai", count=None)
        assert data["query"] == "crewai"
        assert len(data["results"]) == 2
        assert data["results"][0] == {
            "title": "CrewAI docs",
            "url": "https://docs.crewai.com",
            "snippet": "Framework for orchestrating agents.",
            "position": 1,
            "date": "2026-06-01",
        }

    def test_run_passes_count_through(self):
        client = MagicMock()
        client.web_search.return_value = _web_search_response()
        tool = IFlowWebSearchTool(client=client)

        tool.run(query="crewai", count=3)

        client.web_search.assert_called_once_with(query="crewai", count=3)


class TestIFlowImageSearchTool:
    def test_run_returns_normalized_images_json(self):
        client = MagicMock()
        client.image_search.return_value = _image_search_response()
        tool = IFlowImageSearchTool(client=client)

        output = tool.run(query="kittens")
        data = json.loads(output)

        client.image_search.assert_called_once_with(query="kittens", count=None)
        assert data["query"] == "kittens"
        assert data["images"][0]["image_url"] == "https://img.example.com/1.jpg"
        assert data["images"][0]["source_url"] == "https://example.com/page"


class TestIFlowWebFetchTool:
    def test_run_returns_page_content_json(self):
        client = MagicMock()
        client.web_fetch.return_value = _web_fetch_response()
        tool = IFlowWebFetchTool(client=client)

        output = tool.run(url="https://example.com")
        data = json.loads(output)

        client.web_fetch.assert_called_once_with(url="https://example.com")
        assert data["title"] == "Example Domain"
        assert data["content"] == "Example page body."
        assert data["from_cache"] is False


class TestIFlowSearchToolBase:
    def test_missing_api_key_raises_clear_error(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IFLOW_API_KEY", None)
            tool = IFlowWebSearchTool()
            with pytest.raises(ValueError, match="IFLOW_API_KEY"):
                tool.run(query="crewai")

    def test_missing_sdk_raises_install_hint(self):
        tool = IFlowWebSearchTool()
        with patch(
            "crewai_tools.tools.iflow_search_tool.base.IFLOW_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="iflow-search"):
                tool.run(query="crewai")

    def test_declares_package_dependency_and_env_var(self):
        tool = IFlowWebSearchTool()
        assert tool.package_dependencies == ["iflow-search"]
        assert any(env.name == "IFLOW_API_KEY" for env in tool.env_vars)


def test_tools_are_exported_from_crewai_tools():
    import crewai_tools

    assert crewai_tools.IFlowWebSearchTool is IFlowWebSearchTool
    assert crewai_tools.IFlowImageSearchTool is IFlowImageSearchTool
    assert crewai_tools.IFlowWebFetchTool is IFlowWebFetchTool
    assert issubclass(IFlowWebSearchTool, IFlowSearchToolBase)
