from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    ApplicationSelector,
    LegacyClient,
    ToolInfo,
)


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
    response = Mock()
    mock_post.return_value = response
    tool_info = ToolInfo(
        app="github",
        action="create_issue",
        connection_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        description="Create an issue",
        parameters={},
    )

    result = LegacyClient().execute_action(tool_info, arguments)

    assert result is response
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
