from unittest.mock import Mock, patch

import pytest

from crewai_tools.tools.crewai_platform_tools._client import (
    _PlatformToolInfo,
    _PlatformToolsClient,
)


TOOL_INFO = _PlatformToolInfo(
    app="test_app",
    action="test_action",
    connection_id=None,
    description="Test action tool",
    parameters={},
)


@patch.dict(
    "os.environ",
    {
        "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
        "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment_uuid",
    },
    clear=True,
)
@patch("crewai_tools.tools.crewai_platform_tools._client.requests.post")
def test_client_executes_action(mock_post):
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"result": "success"}
    mock_post.return_value = response

    result = _PlatformToolsClient().execute_action(
        TOOL_INFO,
        {"test_param": "test_value"},
    )

    mock_post.assert_called_once_with(
        url=(
            "https://app.crewai.com/clipper/v1/applications/"
            "test_app/tools/test_action/execute"
        ),
        headers={
            "Authorization": "Bearer test_token",
            "X-Crewai-Deployment-Instance-Id": "deployment_uuid",
            "Content-Type": "application/json",
        },
        json={"arguments": {"test_param": "test_value"}},
        timeout=60,
        verify=True,
    )
    assert result is response


@patch.dict(
    "os.environ",
    {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
    clear=True,
)
@patch("crewai_tools.tools.crewai_platform_tools._client.requests.post")
def test_client_returns_action_error_response(mock_post):
    response = Mock(ok=False, status_code=500)
    response.json.return_value = {"error": "Action failed"}
    mock_post.return_value = response

    result = _PlatformToolsClient().execute_action(TOOL_INFO, {})

    assert result is response


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
@patch("crewai_tools.tools.crewai_platform_tools._client.requests.post")
def test_client_ssl_verification(
    mock_post,
    monkeypatch,
    factory_value,
    expected,
):
    monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "test_token")
    if factory_value is None:
        monkeypatch.delenv("CREWAI_FACTORY", raising=False)
    else:
        monkeypatch.setenv("CREWAI_FACTORY", factory_value)
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"result": "success"}
    mock_post.return_value = response

    _PlatformToolsClient().execute_action(TOOL_INFO, {})

    assert mock_post.call_args.kwargs["verify"] is expected
