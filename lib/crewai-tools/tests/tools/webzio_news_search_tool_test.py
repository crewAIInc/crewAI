import os
from unittest.mock import MagicMock, patch

from crewai_tools import WebzioNewsSearchTool
from crewai_tools.tools.webzio_tools.webzio_news_search_tool import (
    DEFAULT_MCP_URL,
    MCP_TOOL_NAME,
    WebzioNewsSearchToolSchema,
)
from pydantic import BaseModel, Field
import pytest


ADAPTER_PATH = "crewai_tools.tools.webzio_tools.webzio_news_search_tool.MCPServerAdapter"
MCP_RESULT = '{"posts": [{"title": "AI Act enters force", "url": "https://example.com"}]}'


class LiveSchema(BaseModel):
    """Stand-in for the argument schema the MCP server advertises."""

    query: str = Field(..., description="The news search query string.")
    language: str | None = Field(None, description="Language filter.")


@pytest.fixture(autouse=True)
def clear_webz_env():
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def mcp_tool():
    tool = MagicMock()
    tool.name = MCP_TOOL_NAME
    tool.args_schema = LiveSchema
    tool._run.return_value = MCP_RESULT
    return tool


@pytest.fixture
def mock_adapter(mcp_tool):
    with patch(ADAPTER_PATH) as adapter_class:
        adapter_class.return_value.tools = [mcp_tool]
        yield adapter_class


def test_connects_with_explicit_token(mock_adapter):
    WebzioNewsSearchTool(api_token="test-token")

    server_params, tool_name = mock_adapter.call_args.args
    assert tool_name == MCP_TOOL_NAME
    assert server_params["url"] == DEFAULT_MCP_URL
    assert server_params["headers"] == {"Authorization": "Bearer test-token"}
    assert mock_adapter.call_args.kwargs == {"connect_timeout": 30}


def test_reads_token_and_url_from_environment(mock_adapter):
    with patch.dict(
        os.environ,
        {"WEBZ_API_TOKEN": "env-token", "WEBZ_MCP_URL": "https://mcp.example.com/mcp/"},
        clear=True,
    ):
        WebzioNewsSearchTool()

    server_params = mock_adapter.call_args.args[0]
    assert server_params["url"] == "https://mcp.example.com/mcp"
    assert server_params["headers"] == {"Authorization": "Bearer env-token"}


def test_adopts_live_args_schema(mock_adapter):
    tool = WebzioNewsSearchTool(api_token="test-token")

    assert tool.args_schema is LiveSchema
    assert "language" in tool.args_schema.model_fields


def test_run_delegates_to_the_mcp_tool(mock_adapter, mcp_tool):
    tool = WebzioNewsSearchTool(api_token="test-token")

    result = tool._run(query="ai regulation", language="english")

    mcp_tool._run.assert_called_once_with(query="ai regulation", language="english")
    assert result == MCP_RESULT


def test_connects_once_across_runs(mock_adapter):
    tool = WebzioNewsSearchTool(api_token="test-token")
    tool._run(query="first")
    tool._run(query="second")

    assert mock_adapter.call_count == 1


def test_construction_survives_an_unreachable_server():
    with patch(ADAPTER_PATH, side_effect=RuntimeError("connection refused")):
        tool = WebzioNewsSearchTool(api_token="test-token")

    assert tool.args_schema is WebzioNewsSearchToolSchema


def test_construction_survives_a_missing_token():
    with patch(ADAPTER_PATH) as adapter_class:
        tool = WebzioNewsSearchTool()

    adapter_class.assert_not_called()
    assert tool.args_schema is WebzioNewsSearchToolSchema


def test_run_reports_a_missing_token():
    tool = WebzioNewsSearchTool()

    with pytest.raises(ValueError, match="WEBZ_API_TOKEN"):
        tool._run(query="ai regulation")


def test_run_retries_a_deferred_connection(mock_adapter, mcp_tool):
    with patch(ADAPTER_PATH, side_effect=RuntimeError("connection refused")):
        tool = WebzioNewsSearchTool(api_token="test-token")

    tool._run(query="ai regulation")

    assert mock_adapter.call_count == 1
    mcp_tool._run.assert_called_once_with(query="ai regulation")


def test_run_reports_a_server_without_the_news_search_tool():
    with patch(ADAPTER_PATH) as adapter_class:
        adapter_class.return_value.tools = []
        tool = WebzioNewsSearchTool(api_token="test-token")

        with pytest.raises(ValueError, match=MCP_TOOL_NAME):
            tool._run(query="ai regulation")


def test_stop_closes_the_session_and_a_later_run_reconnects(mock_adapter):
    tool = WebzioNewsSearchTool(api_token="test-token")
    tool.stop()

    mock_adapter.return_value.stop.assert_called_once()

    tool._run(query="ai regulation")
    assert mock_adapter.call_count == 2


def test_stop_is_idempotent(mock_adapter):
    tool = WebzioNewsSearchTool(api_token="test-token")
    tool.stop()
    tool.stop()

    mock_adapter.return_value.stop.assert_called_once()


def test_context_manager_closes_the_session(mock_adapter):
    with WebzioNewsSearchTool(api_token="test-token") as tool:
        assert tool.args_schema is LiveSchema

    mock_adapter.return_value.stop.assert_called_once()


def test_fallback_schema_allows_server_side_filters():
    validated = WebzioNewsSearchToolSchema.model_validate(
        {"query": "ai regulation", "language": "english"}
    )

    assert validated.model_dump() == {"query": "ai regulation", "language": "english"}


def test_appears_in_tool_specs():
    from crewai_tools.generate_tool_specs import ToolSpecExtractor

    specs = {tool["name"]: tool for tool in ToolSpecExtractor().extract_all_tools()}

    assert specs["WebzioNewsSearchTool"]["humanized_name"] == "Webzio News Search"
    assert {env["name"] for env in specs["WebzioNewsSearchTool"]["env_vars"]} == {
        "WEBZ_API_TOKEN",
        "WEBZ_MCP_URL",
    }
