from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from crewai_tools import ScreenContextAgentTool as TopLevelScreenContextAgentTool
from crewai_tools.generate_tool_specs import ToolSpecExtractor
from crewai_tools.tools import ScreenContextAgentTool as ToolsScreenContextAgentTool
from crewai_tools.tools.screen_context_agent_tool.screen_context_agent_tool import (
    ScreenContextAgentTool,
    ScreenContextAgentToolSchema,
)


MCP_ADAPTER = "crewai_tools.adapters.mcp_adapter.MCPServerAdapter"


def test_initialization_does_not_connect_to_screen_context():
    with patch(MCP_ADAPTER) as adapter:
        tool = ScreenContextAgentTool()
        assert tool.args_schema.model_fields.keys() == {
            "query",
            "since_minutes",
            "limit",
        }

    adapter.assert_not_called()


def test_tool_never_caches_history_results():
    tool = ScreenContextAgentTool()

    assert tool.cache_function({}, {"records": []}) is False


def test_since_minutes_uses_upstream_ten_year_bound():
    schema = ScreenContextAgentToolSchema(query="test", since_minutes=5_256_000)

    assert schema.since_minutes == 5_256_000
    assert "ten years" in ScreenContextAgentToolSchema.model_fields[
        "since_minutes"
    ].description
    with pytest.raises(ValidationError):
        ScreenContextAgentToolSchema(query="test", since_minutes=5_256_001)

    with pytest.raises(ValueError, match="since_minutes must be between 1 and 5256000"):
        ScreenContextAgentTool()._run(query="test", since_minutes=5_256_001)


def test_tool_is_exported_and_included_in_tool_specs():
    assert TopLevelScreenContextAgentTool is ScreenContextAgentTool
    assert ToolsScreenContextAgentTool is ScreenContextAgentTool
    names = {spec["name"] for spec in ToolSpecExtractor().extract_all_tools()}

    assert "ScreenContextAgentTool" in names


def test_run_queries_bounded_history_and_preserves_metadata():
    response = {
        "source": "observed_screen",
        "trust": "untrusted",
        "count": 1,
        "records": [
            {
                "text": "The deployment returned an error.",
                "app": "Browser",
                "ts": 1_760_000_000.0,
                "source": "observed_screen",
                "trust": "untrusted",
            }
        ],
    }
    adapter = MagicMock()
    adapter.tools.__getitem__.return_value.run.return_value = response
    with patch(MCP_ADAPTER, return_value=adapter) as make_adapter:
        tool = ScreenContextAgentTool(
            command="/local/screen-context",
            command_args=["serve", "--profile", "standard", "--transport", "stdio"],
            env={"SCREEN_CONTEXT_CLIENT_TOKEN": "client-token"},
        )
        assert make_adapter.call_count == 0

        result = tool.run(query="deployment error", since_minutes=30, limit=5)

    params = make_adapter.call_args.args[0]
    assert params.command == "/local/screen-context"
    assert params.args == ["serve", "--profile", "standard", "--transport", "stdio"]
    assert params.env == {"SCREEN_CONTEXT_CLIENT_TOKEN": "client-token"}
    adapter.tools.__getitem__.return_value.run.assert_called_once_with(
        query="deployment error", since_minutes=30, limit=5
    )
    assert result["records"][0]["text"] == "The deployment returned an error."
    assert result["records"][0]["app"] == "Browser"
    assert result["records"][0]["ts"] == 1_760_000_000.0
    assert result["records"][0]["trust"] == "untrusted"
    adapter.stop.assert_called_once()


def test_empty_results_are_returned_unchanged():
    response = {"source": "observed_screen", "trust": "untrusted", "count": 0, "records": []}
    adapter = MagicMock()
    adapter.tools.__getitem__.return_value.run.return_value = response
    with patch(MCP_ADAPTER, return_value=adapter):
        result = ScreenContextAgentTool()._run(query="no match")

    assert result == response


def test_connection_failure_is_reported_and_adapter_is_closed():
    adapter = MagicMock()
    adapter.tools.__getitem__.return_value.run.side_effect = ConnectionError("offline")
    with patch(MCP_ADAPTER, return_value=adapter):
        with pytest.raises(RuntimeError, match="ScreenContextAgent history search failed: offline"):
            ScreenContextAgentTool()._run(query="test")

    adapter.stop.assert_called_once()


def test_malformed_mcp_response_is_rejected():
    adapter = MagicMock()
    adapter.tools.__getitem__.return_value.run.return_value = "not json"
    with patch(MCP_ADAPTER, return_value=adapter):
        with pytest.raises(ValueError, match="invalid response"):
            ScreenContextAgentTool()._run(query="test")

    adapter.stop.assert_called_once()


def test_limits_follow_screen_context_bounds():
    with pytest.raises(ValueError):
        ScreenContextAgentTool()._run(query="test", limit=51)
    with pytest.raises(ValueError):
        ScreenContextAgentTool()._run(query="test", since_minutes=0)
