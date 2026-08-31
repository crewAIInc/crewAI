from unittest.mock import Mock, patch

import pytest
import requests

from crewai_tools.tools.crewai_platform_tools._client import (
    _PlatformToolInfo,
    _PlatformToolSelector,
    _PlatformToolsClient,
)


@patch.dict(
    "os.environ",
    {
        "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
        "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment_uuid",
    },
    clear=True,
)
@patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
def test_client_resolves_tool_info(mock_get):
    github_response = Mock()
    github_response.raise_for_status.return_value = None
    github_response.json.return_value = {
        "data": [
            {
                "slug": "create_issue",
                "description": "Create a GitHub issue",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                },
            }
        ]
    }
    slack_response = Mock()
    slack_response.raise_for_status.return_value = None
    slack_response.json.return_value = {
        "data": {
            "slug": "send_message",
            "description": "Send a Slack message",
            "input_schema": {},
        }
    }
    mock_get.side_effect = [github_response, slack_response]
    selectors = [
        _PlatformToolSelector.from_string(
            "github@550e8400-e29b-41d4-a716-446655440000"
        ),
        _PlatformToolSelector.from_string(
            "slack/send_message@67e55044-10b1-426f-9247-bb680e5fe0c8"
        ),
    ]

    resolved = _PlatformToolsClient().get_tools(selectors)

    assert [call.kwargs for call in mock_get.call_args_list] == [
        {
            "headers": {
                "Authorization": "Bearer test_token",
                "X-Crewai-Deployment-Instance-Id": "deployment_uuid",
            },
            "timeout": 30,
            "params": {
                "connection_id": "550e8400-e29b-41d4-a716-446655440000"
            },
            "verify": True,
        },
        {
            "headers": {
                "Authorization": "Bearer test_token",
                "X-Crewai-Deployment-Instance-Id": "deployment_uuid",
            },
            "timeout": 30,
            "params": {
                "connection_id": "67e55044-10b1-426f-9247-bb680e5fe0c8"
            },
            "verify": True,
        },
    ]
    assert [call.args[0] for call in mock_get.call_args_list] == [
        "https://app.crewai.com/clipper/v1/applications/github/tools",
        "https://app.crewai.com/clipper/v1/applications/slack/tools/send_message",
    ]
    assert resolved == [
        _PlatformToolInfo(
            app="github",
            action="create_issue",
            connection_id=selectors[0].connection_id,
            description="Create a GitHub issue",
            parameters={
                "type": "object",
                "properties": {"title": {"type": "string"}},
            },
        ),
        _PlatformToolInfo(
            app="slack",
            action="send_message",
            connection_id=selectors[1].connection_id,
            description="Send a Slack message",
            parameters={},
        ),
    ]


@patch.dict(
    "os.environ",
    {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
    clear=True,
)
@patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
def test_client_accepts_empty_action_list(mock_get):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": []}
    mock_get.return_value = response

    resolved = _PlatformToolsClient().get_tools(
        [_PlatformToolSelector.from_string("github")]
    )

    assert resolved == []


@patch.dict(
    "os.environ",
    {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
    clear=True,
)
@patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
def test_client_keeps_tools_when_one_selector_is_unavailable(mock_get):
    github_response = Mock()
    github_response.raise_for_status.return_value = None
    github_response.json.return_value = {
        "data": [
            {
                "slug": "create_issue",
                "description": "Create a GitHub issue",
                "input_schema": {},
            }
        ]
    }
    unavailable_response = Mock()
    unavailable_response.raise_for_status.side_effect = requests.HTTPError(
        "Application unavailable"
    )
    slack_response = Mock()
    slack_response.raise_for_status.return_value = None
    slack_response.json.return_value = {
        "data": [
            {
                "slug": "send_message",
                "description": "Send a Slack message",
                "input_schema": {},
            }
        ]
    }
    mock_get.side_effect = [github_response, unavailable_response, slack_response]

    resolved = _PlatformToolsClient().get_tools(
        [
            _PlatformToolSelector.from_string("github"),
            _PlatformToolSelector.from_string("unavailable"),
            _PlatformToolSelector.from_string("slack"),
        ]
    )

    assert resolved == [
        _PlatformToolInfo(
            app="github",
            action="create_issue",
            connection_id=None,
            description="Create a GitHub issue",
            parameters={},
        ),
        _PlatformToolInfo(
            app="slack",
            action="send_message",
            connection_id=None,
            description="Send a Slack message",
            parameters={},
        ),
    ]


@pytest.mark.parametrize(
    ("factory_value", "expected"),
    [
        (None, True),
        ("false", True),
        ("FALSE", True),
        ("true", False),
        ("TRUE", False),
    ],
)
@patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
def test_client_ssl_verification(mock_get, monkeypatch, factory_value, expected):
    monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "test_token")
    if factory_value is None:
        monkeypatch.delenv("CREWAI_FACTORY", raising=False)
    else:
        monkeypatch.setenv("CREWAI_FACTORY", factory_value)
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": []}
    mock_get.return_value = response

    _PlatformToolsClient().get_tools(
        [_PlatformToolSelector.from_string("github")]
    )

    assert mock_get.call_args.kwargs["verify"] is expected


def test_client_requires_token(monkeypatch):
    monkeypatch.delenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", raising=False)

    with pytest.raises(ValueError, match="No platform integration token found"):
        _PlatformToolsClient().get_tools(
            [_PlatformToolSelector.from_string("github")]
        )


@pytest.mark.parametrize(
    ("selector", "message"),
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
def test_rejects_invalid_selector(selector, message):
    with pytest.raises(ValueError) as error:
        _PlatformToolSelector.from_string(selector)

    assert repr(selector) in str(error.value)
    assert message in str(error.value)
