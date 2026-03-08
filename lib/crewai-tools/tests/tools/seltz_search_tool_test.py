import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_seltz():
    """Mock the seltz module before importing the tool."""
    mock_client_instance = MagicMock()
    mock_seltz_class = MagicMock(return_value=mock_client_instance)
    mock_includes_class = MagicMock()

    with (
        patch.dict(
            "sys.modules",
            {
                "seltz": MagicMock(
                    Seltz=mock_seltz_class,
                    Includes=mock_includes_class,
                ),
            },
        ),
        patch(
            "crewai_tools.tools.seltz_search_tool.seltz_search_tool.SELTZ_AVAILABLE",
            True,
        ),
        patch(
            "crewai_tools.tools.seltz_search_tool.seltz_search_tool.Seltz",
            mock_seltz_class,
        ),
    ):
        yield {
            "client": mock_client_instance,
            "seltz_class": mock_seltz_class,
            "includes_class": mock_includes_class,
        }


@pytest.fixture
def seltz_tool(mock_seltz):
    from crewai_tools.tools.seltz_search_tool.seltz_search_tool import SeltzSearchTool

    tool = SeltzSearchTool(api_key="test-key")
    tool.client = mock_seltz["client"]
    return tool


def test_seltz_tool_initialization(mock_seltz):
    from crewai_tools.tools.seltz_search_tool.seltz_search_tool import SeltzSearchTool

    tool = SeltzSearchTool(api_key="test-key")
    assert tool.max_documents == 5
    assert tool.context is None
    assert tool.profile is None
    assert tool.max_content_length_per_result == 1000
    assert tool.name == "Seltz Web Knowledge Search"


def test_seltz_tool_search(seltz_tool, mock_seltz):
    mock_doc1 = SimpleNamespace(url="https://example.com/1", content="Result content 1")
    mock_doc2 = SimpleNamespace(url="https://example.com/2", content="Result content 2")
    mock_response = SimpleNamespace(documents=[mock_doc1, mock_doc2])
    mock_seltz["client"].search.return_value = mock_response

    result = seltz_tool.run(query="test query")
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["url"] == "https://example.com/1"
    assert data[0]["content"] == "Result content 1"
    assert data[1]["url"] == "https://example.com/2"
    assert data[1]["content"] == "Result content 2"


def test_seltz_tool_custom_params(mock_seltz):
    from crewai_tools.tools.seltz_search_tool.seltz_search_tool import SeltzSearchTool

    tool = SeltzSearchTool(
        api_key="test-key",
        max_documents=10,
        context="AI research background",
        profile="research",
    )
    tool.client = mock_seltz["client"]

    assert tool.max_documents == 10
    assert tool.context == "AI research background"
    assert tool.profile == "research"

    mock_response = SimpleNamespace(
        documents=[SimpleNamespace(url="https://example.com", content="Content")]
    )
    mock_seltz["client"].search.return_value = mock_response

    tool.run(query="test")

    call_kwargs = mock_seltz["client"].search.call_args[1]
    assert call_kwargs["query"] == "test"
    assert call_kwargs["context"] == "AI research background"
    assert call_kwargs["profile"] == "research"


def test_seltz_tool_content_truncation(seltz_tool, mock_seltz):
    long_content = "x" * 2000
    mock_response = SimpleNamespace(
        documents=[SimpleNamespace(url="https://example.com", content=long_content)]
    )
    mock_seltz["client"].search.return_value = mock_response

    result = seltz_tool.run(query="test")
    data = json.loads(result)

    assert len(data[0]["content"]) == 1003  # 1000 + "..."
    assert data[0]["content"].endswith("...")


def test_seltz_tool_error_handling(seltz_tool, mock_seltz):
    mock_seltz["client"].search.side_effect = Exception("API connection failed")

    result = seltz_tool.run(query="test")
    assert "error" in result.lower() or "Error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
