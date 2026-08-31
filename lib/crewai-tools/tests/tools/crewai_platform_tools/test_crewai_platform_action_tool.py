from unittest.mock import Mock, patch
from uuid import UUID

from crewai.tools.tool_failure import ToolFailure
from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    ClipperClient,
    CrewAIPlatformActionTool,
    LegacyIntegrationsClient,
)
from crewai_tools.tools.crewai_platform_tools.platform_tool import PlatformTool
import pytest


@pytest.mark.parametrize(
    ("factory_value", "expected_verify"),
    [
        (None, True),
        ("false", True),
        ("FALSE", True),
        ("true", False),
        ("TRUE", False),
    ],
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
)
def test_legacy_client_uses_factory_ssl_setting_for_execution(
    mock_post, monkeypatch, factory_value, expected_verify
):
    monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "test_token")
    if factory_value is None:
        monkeypatch.delenv("CREWAI_FACTORY", raising=False)
    else:
        monkeypatch.setenv("CREWAI_FACTORY", factory_value)
    response = Mock(ok=True)
    response.json.return_value = {"result": "success"}
    mock_post.return_value = response

    LegacyIntegrationsClient().execute_tool(
        tool=PlatformTool(
            application="test_app",
            tool="test_action",
            description="Test action tool",
            input_schema={},
        ),
        arguments={"test_param": "test_value"},
    )

    assert mock_post.call_args.kwargs["verify"] is expected_verify


@pytest.mark.parametrize(
    ("factory_value", "expected_verify"),
    [
        (None, True),
        ("false", True),
        ("FALSE", True),
        ("true", False),
        ("TRUE", False),
    ],
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
)
def test_legacy_client_uses_factory_ssl_setting_for_discovery(
    mock_get, monkeypatch, factory_value, expected_verify
):
    monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "test_token")
    if factory_value is None:
        monkeypatch.delenv("CREWAI_FACTORY", raising=False)
    else:
        monkeypatch.setenv("CREWAI_FACTORY", factory_value)
    response = Mock()
    response.json.return_value = {"actions": {}}
    mock_get.return_value = response

    LegacyIntegrationsClient().list_tools(PlatformTool.from_selector("github"))

    assert mock_get.call_args.kwargs["verify"] is expected_verify


@patch.dict(
    "os.environ",
    {
        "CREWAI_PLUS_URL": "https://platform.example",
        "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
    },
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
)
def test_legacy_client_resolves_tools_for_requested_app(mock_get):
    response = Mock()
    response.json.return_value = {
        "actions": {
            "github": [
                {
                    "name": "Create Issue",
                    "description": "Create a GitHub issue",
                    "parameters": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                    },
                }
            ]
        }
    }
    mock_get.return_value = response

    tools = LegacyIntegrationsClient().list_tools(
        PlatformTool.from_selector("github")
    )

    assert tools == [
        PlatformTool(
            application="github",
            tool="Create Issue",
            description="Create a GitHub issue",
            input_schema={
                "type": "object",
                "properties": {"title": {"type": "string"}},
            },
        )
    ]
    request = mock_get.call_args
    assert request.args[0] == (
        "https://platform.example/crewai_plus/api/v1/integrations/actions"
    )
    assert request.kwargs["headers"] == {"Authorization": "Bearer test_token"}
    assert request.kwargs["params"] == {"apps": "github"}


def test_action_tool_returns_failure_when_client_execution_raises():
    client = Mock()
    client.execute_tool.side_effect = RuntimeError("provider unavailable")
    tool = CrewAIPlatformActionTool(
        platform_tool=PlatformTool(
            application="slack",
            tool="send_message",
            description="Send a message",
            input_schema={},
        ),
        client=client,
    )

    result = tool.run()

    assert result == ToolFailure(
        message="Error executing action send_message: provider unavailable",
        code="RuntimeError",
        details={"action": "send_message"},
    )


def test_action_tool_preserves_null_arguments():
    platform_tool = PlatformTool(
        application="github",
        tool="create_issue",
        description="Create a GitHub issue",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
    )
    client = Mock()
    client.execute_tool.return_value = {"issue": 42}
    tool = CrewAIPlatformActionTool(platform_tool=platform_tool, client=client)

    result = tool.run(title="Foundational work", body=None)

    assert result == {"issue": 42}
    client.execute_tool.assert_called_once_with(
        platform_tool,
        {"title": "Foundational work", "body": None},
    )


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
)
def test_legacy_client_executes_tool(mock_post):
    response = Mock(ok=True)
    response.json.return_value = {"result": "sent"}
    mock_post.return_value = response

    result = LegacyIntegrationsClient().execute_tool(
        tool=PlatformTool(
            application="slack",
            tool="send_message",
            description="Send a message",
            input_schema={},
        ),
        arguments={"text": "Hello"},
    )

    assert result == '{\n  "result": "sent"\n}'
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"] == {"integration": {"text": "Hello"}}
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer test_token"
    )


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
)
def test_legacy_client_maps_api_error(mock_post):
    response = Mock(ok=False, status_code=503)
    response.json.return_value = {"error": {"message": "Service unavailable"}}
    mock_post.return_value = response

    result = LegacyIntegrationsClient().execute_tool(
        tool=PlatformTool(
            application="slack",
            tool="send_message",
            description="Send a message",
            input_schema={},
        ),
        arguments={},
    )

    assert isinstance(result, ToolFailure)
    assert result.message == "API request failed: Service unavailable"
    assert result.code == "503"
    assert result.retryable is True
    assert mock_post.call_args.kwargs["json"] == {"integration": {"_noop": True}}


@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
)
def test_clipper_client_resolves_selector(mock_get):
    connection_id = "123e4567-e89b-12d3-a456-426614174000"
    schema_response = Mock()
    schema_response.json.return_value = {
        "data": {
            "slug": "send-email",
            "description": "Send an email",
            "input_schema": {
                "type": "object",
                "properties": {"recipient": {"type": "string"}},
            },
        }
    }
    mock_get.return_value = schema_response
    client = ClipperClient(
        integration_token="token",
        deployment_instance_uuid="deployment-uuid",
        base_url="https://example.test",
    )

    definitions = client.list_tools(
        PlatformTool.from_selector(f"gmail/send-email@{connection_id}")
    )

    assert len(definitions) == 1
    assert definitions[0].application == "gmail"
    assert definitions[0].python_identifier == "gmail_send_email"
    assert definitions[0].tool == "send-email"
    assert definitions[0].connection_id == UUID(connection_id)
    assert definitions[0].input_schema == {
        "type": "object",
        "properties": {"recipient": {"type": "string"}},
    }
    schema_call = mock_get.call_args
    assert schema_call.args[0] == (
        "https://example.test/clipper/v1/applications/gmail/tools/send-email"
    )
    assert schema_call.kwargs["params"] == {"connection_id": connection_id}
    assert mock_get.call_count == 1


@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
)
def test_clipper_client_does_not_translate_tool_slug(mock_get):
    response = Mock()
    response.json.return_value = {
        "data": {
            "slug": "send_email",
            "description": "Send an email",
            "input_schema": {},
        }
    }
    mock_get.return_value = response
    client = ClipperClient(
        integration_token="token",
        deployment_instance_uuid="deployment-uuid",
    )

    definitions = client.list_tools(PlatformTool.from_selector("gmail/send_email"))

    assert definitions[0].tool == "send_email"
    assert mock_get.call_args.args[0].endswith(
        "/clipper/v1/applications/gmail/tools/send_email"
    )


@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
)
def test_clipper_client_resolves_all_tools_for_application(mock_get):
    connection_id = "123e4567-e89b-12d3-a456-426614174000"
    response = Mock()
    response.json.return_value = {
        "data": [
            {
                "slug": "send-email",
                "description": "Send an email",
                "input_schema": {"type": "object"},
            },
            {
                "slug": "list-drafts",
                "description": "List email drafts",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
    }
    mock_get.return_value = response
    client = ClipperClient(
        integration_token="token",
        base_url="https://example.test",
    )

    definitions = client.list_tools(
        PlatformTool.from_selector(f"gmail@{connection_id}")
    )

    assert definitions == [
        PlatformTool(
            application="gmail",
            tool="send-email",
            description="Send an email",
            input_schema={"type": "object"},
            connection_id=UUID(connection_id),
        ),
        PlatformTool(
            application="gmail",
            tool="list-drafts",
            description="List email drafts",
            input_schema={"type": "object", "properties": {}},
            connection_id=UUID(connection_id),
        ),
    ]
    call = mock_get.call_args
    assert call.args[0] == (
        "https://example.test/clipper/v1/applications/gmail/tools"
    )
    assert call.kwargs["params"] == {"connection_id": connection_id}


@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
)
def test_clipper_client_executes_tool(mock_post):
    connection_id = "123e4567-e89b-12d3-a456-426614174000"
    response = Mock(ok=True)
    response.json.return_value = {"data": {"output": {"message_id": "123"}}}
    mock_post.return_value = response
    client = ClipperClient(
        integration_token="token",
        deployment_instance_uuid="deployment-uuid",
        base_url="https://example.test/",
    )

    result = client.execute_tool(
        tool=PlatformTool(
            application="gmail",
            tool="send-email",
            description="Send an email",
            input_schema={},
            connection_id=UUID(connection_id),
        ),
        arguments={"recipient": "member@example.com"},
    )

    assert result == '{\n  "message_id": "123"\n}'
    call = mock_post.call_args
    assert call.kwargs["url"] == (
        "https://example.test/clipper/v1/applications/gmail/tools/send-email/execute"
    )
    assert call.kwargs["json"] == {
        "arguments": {"recipient": "member@example.com"},
        "connection_id": connection_id,
    }
    assert call.kwargs["headers"]["Authorization"] == "Bearer token"
    assert (
        call.kwargs["headers"]["X-Crewai-Deployment-Instance-Id"] == "deployment-uuid"
    )
    assert (
        str(UUID(call.kwargs["headers"]["Idempotency-Key"]))
        == (call.kwargs["headers"]["Idempotency-Key"])
    )


@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
)
def test_clipper_client_maps_problem(mock_post):
    response = Mock(ok=False, status_code=422)
    response.json.return_value = {
        "errors": [
            {
                "code": "tool_execution_failed",
                "detail": "The provider could not complete the tool call.",
            }
        ]
    }
    mock_post.return_value = response
    client = ClipperClient(
        integration_token="token",
        deployment_instance_uuid="deployment-uuid",
    )

    result = client.execute_tool(
        tool=PlatformTool(
            application="gmail",
            tool="send-email",
            description="Send an email",
            input_schema={},
        ),
        arguments={},
    )

    assert isinstance(result, ToolFailure)
    assert result.message == "The provider could not complete the tool call."
    assert result.code == "tool_execution_failed"
    assert result.retryable is False


@patch.dict("os.environ", {}, clear=True)
@patch(
    "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
)
def test_clipper_client_omits_missing_deployment_identity(mock_get):
    response = Mock()
    response.json.return_value = {
        "data": {
            "slug": "send-email",
            "description": "Send an email",
            "input_schema": {},
        }
    }
    mock_get.return_value = response
    client = ClipperClient(integration_token="token")

    definitions = client.list_tools(PlatformTool.from_selector("gmail/send-email"))

    assert len(definitions) == 1
    assert (
        "X-Crewai-Deployment-Instance-Id" not in (mock_get.call_args.kwargs["headers"])
    )
