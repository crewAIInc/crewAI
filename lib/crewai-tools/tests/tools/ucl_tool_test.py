from unittest.mock import patch, MagicMock

import pytest

from crewai_tools.tools.ucl_tool.ucl_tool import UCLTool, UCLToolConfig


@pytest.fixture
def ucl_config():
    return UCLToolConfig(
        workspace_id="test-workspace-id",
        api_key="test-api-key",
        mcp_gateway_id="test-mcp-gateway-id",
    )


def test_ucl_config_creation():
    config = UCLTool.from_config(
        workspace_id="test-workspace",
        api_key="test-key",
        mcp_gateway_id="test-gateway",
    )
    assert config.workspace_id == "test-workspace"
    assert config.api_key == "test-key"
    assert config.mcp_gateway_id == "test-gateway"
    assert config.base_url == "https://live.fastn.ai"
    assert config.stage == "LIVE"


def test_ucl_config_custom_base_url():
    config = UCLTool.from_config(
        workspace_id="test-workspace",
        api_key="test-key",
        mcp_gateway_id="test-gateway",
        base_url="https://custom.api.com",
        stage="DEV",
    )
    assert config.base_url == "https://custom.api.com"
    assert config.stage == "DEV"


def test_get_headers(ucl_config):
    headers = UCLTool._get_headers(ucl_config)
    assert headers["stage"] == "LIVE"
    assert headers["x-fastn-space-id"] == "test-workspace-id"
    assert headers["x-fastn-api-key"] == "test-api-key"
    assert headers["x-fastn-space-agent-id"] == "test-mcp-gateway-id"
    assert headers["Content-Type"] == "application/json"


@patch("requests.post")
def test_fetch_tools(mock_post, ucl_config):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "actionId": "action-123",
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                    },
                    "required": ["to", "subject"],
                },
            },
        }
    ]
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    tools_data = UCLTool._fetch_tools(ucl_config, prompt="email", limit=10)

    assert len(tools_data) == 1
    assert tools_data[0]["actionId"] == "action-123"
    assert tools_data[0]["function"]["name"] == "send_email"

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"] == {"input": {"limit": 10, "prompt": "email"}}


@patch("requests.post")
def test_execute_tool(mock_post, ucl_config):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "body": {"success": True, "message": "Email sent"},
        "statusCode": 200,
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = UCLTool._execute_tool(
        config=ucl_config,
        action_id="action-123",
        tool_name="send_email",
        parameters={"to": "test@example.com", "subject": "Test"},
    )

    assert result == {"success": True, "message": "Email sent"}
    mock_post.assert_called_once()


@patch("requests.post")
def test_get_tools(mock_post, ucl_config):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "actionId": "action-123",
            "type": "function",
            "function": {
                "name": "send_slack_message",
                "description": "Send a message to Slack",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "description": "Channel name"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["channel", "message"],
                },
            },
        },
        {
            "actionId": "action-456",
            "type": "function",
            "function": {
                "name": "get_slack_channels",
                "description": "List Slack channels",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    tools = UCLTool.get_tools(ucl_config, prompt="slack", limit=10)

    assert len(tools) == 2
    assert tools[0].name == "send_slack_message"
    assert tools[0].description == "Send a message to Slack"
    assert tools[1].name == "get_slack_channels"


def test_from_action(ucl_config):
    tool = UCLTool.from_action(
        config=ucl_config,
        action_id="test-action",
        tool_name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
            },
            "required": ["param1"],
        },
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.action_id == "test-action"


def test_json_schema_to_pydantic_model():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User name"},
            "age": {"type": "integer", "description": "User age"},
            "active": {"type": "boolean", "description": "Is active"},
        },
        "required": ["name"],
    }

    model = UCLTool._json_schema_to_pydantic_model(schema, "TestModel")

    assert model.__name__ == "TestModel"
    fields = model.model_fields
    assert "name" in fields
    assert "age" in fields
    assert "active" in fields


@patch("requests.post")
def test_tool_run(mock_post, ucl_config):
    # Mock the execute response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "body": {"result": "success"},
        "statusCode": 200,
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    tool = UCLTool.from_action(
        config=ucl_config,
        action_id="test-action",
        tool_name="test_tool",
        description="A test tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
            },
            "required": ["param1"],
        },
    )

    result = tool._run(param1="test_value")
    assert result == {"result": "success"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
