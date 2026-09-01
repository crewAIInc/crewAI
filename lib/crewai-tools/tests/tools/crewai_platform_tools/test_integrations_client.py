from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import Mock, call, patch
from uuid import UUID

import pytest
from requests.exceptions import JSONDecodeError

from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    ApplicationSelector,
    ClipperClient,
    IntegrationsClient,
    LegacyClient,
    ToolExecutionFailure,
    ToolExecutionSuccess,
    ToolInfo,
)


@patch.dict(
    "os.environ",
    {
        "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
        "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment-instance-id",
        "CREWAI_FACTORY": "false",
        "CREWAI_PLUS_URL": "https://platform.example.test/",
    },
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
)
def test_clipper_client_discovers_selected_actions(mock_get: Mock) -> None:
    index_response = Mock()
    index_response.raise_for_status.return_value = None
    index_response.json.return_value = {
        "data": [
            {
                "slug": "create_issue",
                "description": "Create a GitHub issue",
                "input_schema": {"type": "object"},
            }
        ]
    }
    show_response = Mock()
    show_response.raise_for_status.return_value = None
    show_response.json.return_value = {
        "data": {
            "slug": "create_issue",
            "description": "Create a GitHub issue",
            "input_schema": {"type": "object"},
        }
    }
    mock_get.side_effect = [index_response, show_response]
    connection_id = UUID("550e8400-e29b-41d4-a716-446655440000")

    tools = ClipperClient().get_actions(
        [
            ApplicationSelector.from_string(f"github@{connection_id}"),
            ApplicationSelector.from_string("github/create_issue"),
        ]
    )

    expected_tool = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=connection_id,
        description="Create a GitHub issue",
        parameters={"type": "object"},
    )
    assert tools == [
        expected_tool,
        ToolInfo(
            app="github",
            action="create_issue",
            connection_id=None,
            description="Create a GitHub issue",
            parameters={"type": "object"},
        ),
    ]
    headers = {
        "Authorization": "Bearer test_token",
        "X-Crewai-Deployment-Instance-Id": "deployment-instance-id",
    }
    assert mock_get.call_args_list == [
        call(
            "https://platform.example.test/clipper/v1/applications/github/tools",
            headers=headers,
            params={"connection_id": str(connection_id)},
            timeout=30,
            verify=True,
        ),
        call(
            "https://platform.example.test/clipper/v1/applications/github/tools/create_issue",
            headers=headers,
            params={},
            timeout=30,
            verify=True,
        ),
    ]
    index_response.raise_for_status.assert_called_once_with()
    show_response.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize(
    ("factory_value", "verify"),
    [
        (None, True),
        ("false", True),
        ("FALSE", True),
        ("true", False),
        ("TRUE", False),
    ],
)
@patch.dict(
    "os.environ",
    {
        "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
        "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment-instance-id",
    },
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
)
def test_clipper_client_preserves_discovery_ssl_behavior(
    mock_get: Mock,
    factory_value: str | None,
    verify: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if factory_value is None:
        monkeypatch.delenv("CREWAI_FACTORY", raising=False)
    else:
        monkeypatch.setenv("CREWAI_FACTORY", factory_value)
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": []}
    mock_get.return_value = response

    ClipperClient().get_actions([ApplicationSelector.from_string("github")])

    assert mock_get.call_args.kwargs["verify"] is verify


@patch.dict(
    "os.environ",
    {
        "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
        "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment-instance-id",
        "CREWAI_FACTORY": "false",
        "CREWAI_PLUS_URL": "https://platform.example.test/",
    },
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
@pytest.mark.parametrize(
    ("arguments", "connection_id"),
    [
        ({}, UUID("550e8400-e29b-41d4-a716-446655440000")),
        (
            {
                "filters": {"labels": ["urgent"], "enabled": True},
                "values": [1, {"key": "value"}],
            },
            None,
        ),
    ],
)
def test_clipper_client_executes_action(
    mock_post: Mock,
    arguments: dict[str, Any],
    connection_id: UUID | None,
) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"data": {"output": {"issue": 42}}}
    mock_post.return_value = response
    tool = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=connection_id,
        description="Create an issue",
        parameters={},
    )

    result = ClipperClient().execute_action(tool, arguments)

    assert result == ToolExecutionSuccess(output={"issue": 42})
    expected_payload: dict[str, Any] = {"arguments": arguments}
    if connection_id is not None:
        expected_payload["connection_id"] = str(connection_id)
    mock_post.assert_called_once_with(
        "https://platform.example.test/clipper/v1/applications/github/tools/create_issue/execute",
        headers={
            "Authorization": "Bearer test_token",
            "X-Crewai-Deployment-Instance-Id": "deployment-instance-id",
        },
        json=expected_payload,
        timeout=60,
        verify=True,
    )


@patch.dict(
    "os.environ",
    {
        "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
        "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment-instance-id",
        "CREWAI_FACTORY": "false",
        "CREWAI_PLUS_URL": "https://platform.example.test/",
    },
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
@pytest.mark.parametrize(
    ("status_code", "code", "detail", "retryable"),
    [
        (
            422,
            "tool_execution_failed",
            "The provider rejected the request.",
            False,
        ),
        (
            503,
            "service_unavailable",
            "The tool provider is unavailable.",
            True,
        ),
    ],
)
def test_clipper_client_normalizes_execution_failure(
    mock_post: Mock,
    status_code: int,
    code: str,
    detail: str,
    retryable: bool,
) -> None:
    response = Mock(status_code=status_code)
    response.json.return_value = {
        "errors": [
            {
                "code": code,
                "detail": detail,
            }
        ]
    }
    mock_post.return_value = response
    tool = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=None,
        description="Create an issue",
        parameters={},
    )

    result = ClipperClient().execute_action(tool, {"title": "Contract test"})

    assert result == ToolExecutionFailure(
        message=detail,
        code=code,
        retryable=retryable,
    )


@patch.dict(
    "os.environ",
    {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
    clear=True,
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
def test_clipper_client_normalizes_non_json_service_failure(
    mock_post: Mock,
) -> None:
    response = Mock(status_code=503)
    response.json.side_effect = JSONDecodeError("Expecting value", "", 0)
    mock_post.return_value = response
    tool = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=None,
        description="Create an issue",
        parameters={},
    )

    result = ClipperClient().execute_action(tool, {"title": "Contract test"})

    assert result == ToolExecutionFailure(
        message="Upstream API request failed with status 503.",
        code="503",
        retryable=True,
    )


@patch.dict(
    "os.environ",
    {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
    clear=True,
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
)
def test_clipper_client_discovers_without_deployment_instance_uuid(
    mock_get: Mock,
) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": []}
    mock_get.return_value = response

    assert ClipperClient().get_actions(
        [ApplicationSelector.from_string("github")]
    ) == []
    assert mock_get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer test_token"
    }


@patch.dict(
    "os.environ",
    {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
    clear=True,
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
def test_clipper_client_executes_without_deployment_instance_uuid(
    mock_post: Mock,
) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"data": {"output": {"issue": 42}}}
    mock_post.return_value = response
    tool = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=None,
        description="Create an issue",
        parameters={},
    )

    result = ClipperClient().execute_action(tool, {})

    assert result == ToolExecutionSuccess(output={"issue": 42})
    assert mock_post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer test_token"
    }


@patch.dict(
    "os.environ",
    {"CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment-instance-id"},
    clear=True,
)
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
def test_clipper_client_requires_platform_integration_token(
    mock_post: Mock,
) -> None:
    tool = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=None,
        description="Create an issue",
        parameters={},
    )

    with pytest.raises(ValueError, match="CREWAI_PLATFORM_INTEGRATION_TOKEN"):
        ClipperClient().execute_action(tool, {})

    mock_post.assert_not_called()


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
)
def test_legacy_client_normalizes_discovered_actions(mock_get: Mock) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "actions": {
            "github": [
                {
                    "name": "create_issue",
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
    connection_id = UUID("550e8400-e29b-41d4-a716-446655440000")

    tools = LegacyClient().get_actions(
        [ApplicationSelector.from_string(f"github/create_issue@{connection_id}")]
    )

    assert tools == [
        ToolInfo(
            app="github",
            action="create_issue",
            connection_id=connection_id,
            description="Create a GitHub issue",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}},
            },
        )
    ]
    response.raise_for_status.assert_called_once_with()
    assert mock_get.call_args.kwargs["params"] == {"apps": "github/create_issue"}


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
)
def test_legacy_client_emits_action_for_each_matching_selector(
    mock_get: Mock,
) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "actions": {
            "github": [
                {
                    "name": "create_issue",
                    "description": "Create a GitHub issue",
                    "parameters": {},
                }
            ]
        }
    }
    mock_get.return_value = response
    app_connection_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    action_connection_id = UUID("8c5f9d69-902b-4b48-a23c-8d037c242e1e")

    tools = LegacyClient().get_actions(
        [
            ApplicationSelector.from_string(f"github@{app_connection_id}"),
            ApplicationSelector.from_string(
                f"github/create_issue@{action_connection_id}"
            ),
        ]
    )

    assert [tool.connection_id for tool in tools] == [
        app_connection_id,
        action_connection_id,
    ]
    assert [tool.qualified_name for tool in tools] == [
        "github_create_issue_550e8400_e29b_41d4_a716_446655440000",
        "github_create_issue_8c5f9d69_902b_4b48_a23c_8d037c242e1e",
    ]


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
)
def test_legacy_client_excludes_actions_without_a_matching_selector(
    mock_get: Mock,
) -> None:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "actions": {
            "github": [
                {
                    "name": "delete_issue",
                    "description": "Delete a GitHub issue",
                    "parameters": {},
                }
            ]
        }
    }
    mock_get.return_value = response

    tools = LegacyClient().get_actions(
        [ApplicationSelector.from_string("github/create_issue")]
    )

    assert tools == []


def test_tool_info_is_immutable() -> None:
    tool_info = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=None,
        description="Create an issue",
        parameters={},
    )

    with pytest.raises(FrozenInstanceError):
        tool_info.action = "delete_issue"


@pytest.mark.parametrize(
    ("result", "field", "value"),
    [
        (ToolExecutionSuccess(output={"issue": 42}), "output", {"issue": 43}),
        (
            ToolExecutionFailure(
                message="Request failed", code="400", retryable=False
            ),
            "message",
            "Another failure",
        ),
    ],
)
def test_tool_execution_results_are_immutable(
    result: ToolExecutionSuccess | ToolExecutionFailure,
    field: str,
    value: Any,
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, value)


def test_application_selector_is_immutable() -> None:
    selector = ApplicationSelector.from_string(
        "github/create_issue@550e8400-e29b-41d4-a716-446655440000"
    )

    assert selector.app == "github"
    assert selector.action == "create_issue"
    assert selector.connection_id == UUID("550e8400-e29b-41d4-a716-446655440000")
    with pytest.raises(FrozenInstanceError):
        selector.action = "delete_issue"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "cannot be empty"),
        (
            "@550e8400-e29b-41d4-a716-446655440000",
            "application cannot be empty",
        ),
        ("github/", "action cannot be empty"),
        ("github@", "connection ID cannot be empty"),
        ("github@not-a-uuid", "connection ID must be a valid UUID"),
        (
            "github@550e8400-e29b-41d4-a716-446655440000/issues",
            "connection ID must be the last segment",
        ),
    ],
)
def test_application_selector_rejects_invalid_values(
    value: str, message: str
) -> None:
    with pytest.raises(ValueError) as error:
        ApplicationSelector.from_string(value)

    assert repr(value) in str(error.value)
    assert message in str(error.value)


@pytest.mark.parametrize(
    ("factory_value", "verify"),
    [
        (None, True),
        ("false", True),
        ("FALSE", True),
        ("true", False),
        ("TRUE", False),
    ],
)
@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
)
def test_legacy_client_preserves_discovery_ssl_behavior(
    mock_get: Mock,
    factory_value: str | None,
    verify: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if factory_value is None:
        monkeypatch.delenv("CREWAI_FACTORY", raising=False)
    else:
        monkeypatch.setenv("CREWAI_FACTORY", factory_value)
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"actions": {}}
    mock_get.return_value = response

    LegacyClient().get_actions([ApplicationSelector.from_string("github")])

    assert mock_get.call_args.kwargs["verify"] is verify


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
@pytest.mark.parametrize(
    ("arguments", "integration"),
    [({"title": "Contract test"}, {"title": "Contract test"}), ({}, {"_noop": True})],
)
def test_legacy_client_preserves_execution_request(
    mock_post: Mock,
    arguments: dict[str, Any],
    integration: dict[str, Any],
) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"issue": 42}
    mock_post.return_value = response
    tool_info = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        description="Create an issue",
        parameters={},
    )

    client: IntegrationsClient = LegacyClient()
    result = client.execute_action(tool_info, arguments)

    assert result == ToolExecutionSuccess(output={"issue": 42})
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["url"].endswith(
        "/actions/create_issue/execute"
    )
    assert mock_post.call_args.kwargs["headers"] == {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json",
    }
    assert mock_post.call_args.kwargs["json"] == {"integration": integration}
    assert mock_post.call_args.kwargs["timeout"] == 60
    assert mock_post.call_args.kwargs["allow_redirects"] is False


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
@pytest.mark.parametrize(
    ("response_data", "status_code", "message", "retryable"),
    [
        ({"error": {"message": "Invalid issue"}}, 400, "Invalid issue", False),
        ({"error": "Rate limited"}, 429, "Rate limited", False),
        (["Service unavailable"], 503, "['Service unavailable']", True),
        ({"reason": "Unknown"}, 500, '{"reason": "Unknown"}', True),
    ],
)
def test_legacy_client_normalizes_execution_failures(
    mock_post: Mock,
    response_data: Any,
    status_code: int,
    message: str,
    retryable: bool,
) -> None:
    response = Mock(status_code=status_code)
    response.json.return_value = response_data
    mock_post.return_value = response
    tool_info = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=None,
        description="Create an issue",
        parameters={},
    )

    result = LegacyClient().execute_action(tool_info, {"title": "Contract test"})

    assert result == ToolExecutionFailure(
        message=message,
        code=str(status_code),
        retryable=retryable,
    )


@patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
@patch(
    "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
)
def test_legacy_client_treats_redirect_as_execution_failure(
    mock_post: Mock,
) -> None:
    response = Mock(status_code=302)
    response.json.return_value = {"error": {"message": "Redirected"}}
    mock_post.return_value = response
    tool_info = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=None,
        description="Create an issue",
        parameters={},
    )

    result = LegacyClient().execute_action(tool_info, {})

    assert result == ToolExecutionFailure(
        message="Redirected",
        code="302",
        retryable=False,
    )
