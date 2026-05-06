from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crewai_tools import SnowflakeCortexAgentTool


@pytest.fixture
def agent_object_tool():
    return SnowflakeCortexAgentTool(
        account="myorg-myaccount",
        auth_token="test-token",
        database="MY_DB",
        snowflake_schema="MY_SCHEMA",
        agent_name="SALES_AGENT",
    )


@pytest.fixture
def inline_tool():
    return SnowflakeCortexAgentTool(
        account="myorg-myaccount",
        auth_token="test-token",
        tools=[
            {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst_tool"}},
            {"tool_spec": {"type": "cortex_search", "name": "search_tool"}},
        ],
        tool_resources={
            "analyst_tool": {"semantic_model_file": "@MY_DB.MY_SCHEMA.MODELS/sales.yaml"},
            "search_tool": {"name": "MY_DB.MY_SCHEMA.MY_SEARCH_SVC"},
        },
        tool_choice={"type": "auto"},
        models={"orchestration": "claude-4-sonnet"},
        instructions={"response": "Be concise."},
    )


def _ok_response(payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload if payload is not None else {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The total revenue for 2025 was $100,000."}
        ],
    }
    response.text = ""
    return response


def test_requires_token_when_env_missing(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_CORTEX_AGENT_TOKEN", raising=False)
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    with pytest.raises(ValueError, match="bearer token"):
        SnowflakeCortexAgentTool(
            account="myorg-myaccount",
            database="MY_DB",
            snowflake_schema="MY_SCHEMA",
            agent_name="SALES_AGENT",
        )


def test_requires_host_or_account(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    with pytest.raises(ValueError, match="account"):
        SnowflakeCortexAgentTool(
            auth_token="test-token",
            database="MY_DB",
            snowflake_schema="MY_SCHEMA",
            agent_name="SALES_AGENT",
        )


def test_partial_agent_object_config_is_rejected():
    with pytest.raises(ValueError, match="agent object"):
        SnowflakeCortexAgentTool(
            account="myorg-myaccount",
            auth_token="test-token",
            database="MY_DB",
            snowflake_schema="MY_SCHEMA",
            # agent_name missing
        )


def test_requires_tools_when_no_agent_object():
    with pytest.raises(ValueError, match="tools"):
        SnowflakeCortexAgentTool(
            account="myorg-myaccount",
            auth_token="test-token",
        )


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "envorg-envaccount")
    monkeypatch.setenv("SNOWFLAKE_CORTEX_AGENT_TOKEN", "env-token")
    tool = SnowflakeCortexAgentTool(
        database="MY_DB",
        snowflake_schema="MY_SCHEMA",
        agent_name="SALES_AGENT",
    )
    assert tool._build_url() == (
        "https://envorg-envaccount.snowflakecomputing.com"
        "/api/v2/databases/MY_DB/schemas/MY_SCHEMA/agents/SALES_AGENT:run"
    )
    assert tool._build_headers()["Authorization"] == "Bearer env-token"


def test_agent_object_url(agent_object_tool):
    assert agent_object_tool._build_url() == (
        "https://myorg-myaccount.snowflakecomputing.com"
        "/api/v2/databases/MY_DB/schemas/MY_SCHEMA/agents/SALES_AGENT:run"
    )


def test_inline_url(inline_tool):
    assert inline_tool._build_url() == (
        "https://myorg-myaccount.snowflakecomputing.com/api/v2/cortex/agent:run"
    )


def test_custom_host_overrides_account():
    tool = SnowflakeCortexAgentTool(
        account="myorg-myaccount",
        host="my-private-link.example.com",
        auth_token="test-token",
        database="MY_DB",
        snowflake_schema="MY_SCHEMA",
        agent_name="SALES_AGENT",
    )
    assert tool._build_url().startswith(
        "https://my-private-link.example.com/api/v2/databases/MY_DB"
    )


def test_custom_host_with_scheme_is_preserved():
    tool = SnowflakeCortexAgentTool(
        host="https://my-private-link.example.com",
        auth_token="test-token",
        database="MY_DB",
        snowflake_schema="MY_SCHEMA",
        agent_name="SALES_AGENT",
    )
    assert tool._build_url() == (
        "https://my-private-link.example.com"
        "/api/v2/databases/MY_DB/schemas/MY_SCHEMA/agents/SALES_AGENT:run"
    )


def test_payload_for_agent_object(agent_object_tool):
    payload = agent_object_tool._build_payload("What is revenue?")
    assert payload["stream"] is False
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is revenue?"}],
        }
    ]
    assert "tools" not in payload
    assert "tool_resources" not in payload


def test_payload_for_inline_tools(inline_tool):
    payload = inline_tool._build_payload("Hello")
    assert payload["tools"][0]["tool_spec"]["name"] == "analyst_tool"
    assert payload["tool_resources"]["search_tool"]["name"] == (
        "MY_DB.MY_SCHEMA.MY_SEARCH_SVC"
    )
    assert payload["tool_choice"] == {"type": "auto"}
    assert payload["models"] == {"orchestration": "claude-4-sonnet"}
    assert payload["instructions"] == {"response": "Be concise."}


def test_run_extracts_text_response(agent_object_tool):
    expected_url = agent_object_tool._build_url()
    response = _ok_response()
    with patch.object(
        agent_object_tool._session, "post", return_value=response
    ) as mock_post:
        result = agent_object_tool._run(query="What is revenue?")
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == expected_url
    assert kwargs["json"]["messages"][0]["content"][0]["text"] == "What is revenue?"
    assert kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert result == "The total revenue for 2025 was $100,000."


def test_run_concatenates_multiple_text_items(agent_object_tool):
    payload = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "First."},
            {"type": "tool_use", "name": "analyst_tool"},
            {"type": "text", "text": "Second."},
        ],
    }
    with patch.object(
        agent_object_tool._session, "post", return_value=_ok_response(payload)
    ):
        result = agent_object_tool._run(query="Hi")
    assert result == "First.\nSecond."


def test_run_returns_full_json_when_no_text_content(agent_object_tool):
    payload = {
        "role": "assistant",
        "content": [{"type": "tool_use", "name": "analyst_tool"}],
    }
    with patch.object(
        agent_object_tool._session, "post", return_value=_ok_response(payload)
    ):
        result = agent_object_tool._run(query="Hi")
    assert "tool_use" in result
    assert "analyst_tool" in result


def test_run_handles_http_error(agent_object_tool):
    response = MagicMock()
    response.status_code = 401
    response.text = "Invalid token"
    response.json.return_value = {}
    with patch.object(agent_object_tool._session, "post", return_value=response):
        result = agent_object_tool._run(query="Hi")
    assert result.startswith("Snowflake Cortex Agent returned HTTP 401")
    assert "Invalid token" in result


def test_run_handles_request_exception(agent_object_tool):
    import requests

    with patch.object(
        agent_object_tool._session,
        "post",
        side_effect=requests.ConnectionError("boom"),
    ):
        result = agent_object_tool._run(query="Hi")
    assert result.startswith("Error calling Snowflake Cortex Agent")
    assert "boom" in result


def test_tool_is_exported_from_top_level():
    import crewai_tools

    assert hasattr(crewai_tools, "SnowflakeCortexAgentTool")
    assert hasattr(crewai_tools, "SnowflakeCortexAgentToolInput")
