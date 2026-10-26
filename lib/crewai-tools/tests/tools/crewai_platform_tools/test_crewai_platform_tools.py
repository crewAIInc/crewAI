import unittest
from unittest.mock import Mock, patch

from crewai.tools import ToolExecutionFailedError, ToolFailure
import pytest

from crewai_tools.tools.crewai_platform_tools import (
    CrewaiPlatformTools,
    run_platform_action,
)


class TestCrewaiPlatformTools(unittest.TestCase):
    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_crewai_platform_tools_basic(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {"github": []}}
        mock_get.return_value = mock_response

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert type(tools) is list

    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_crewai_platform_tools_multiple_apps(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create a GitHub issue",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Issue title",
                                },
                                "body": {"type": "string", "description": "Issue body"},
                            },
                            "required": ["title"],
                        },
                    }
                ],
                "slack": [
                    {
                        "name": "send_message",
                        "description": "Send a Slack message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "channel": {
                                    "type": "string",
                                    "description": "Channel to send to",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Message text",
                                },
                            },
                            "required": ["channel", "text"],
                        },
                    }
                ],
            }
        }
        github_response = Mock()
        github_response.raise_for_status.return_value = None
        github_response.json.return_value = {
            "actions": {"github": mock_response.json.return_value["actions"]["github"]}
        }
        slack_response = Mock()
        slack_response.raise_for_status.return_value = None
        slack_response.json.return_value = {
            "actions": {"slack": mock_response.json.return_value["actions"]["slack"]}
        }
        mock_get.side_effect = [github_response, slack_response]

        tools = CrewaiPlatformTools(apps=["github", "slack"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 2
        assert [tool.name for tool in tools] == [
            "github_create_issue",
            "slack_send_message",
        ]
        assert [tool.app for tool in tools] == ["github", "slack"]
        assert tools[0].description == "Create a GitHub issue"
        assert tools[1].description == "Send a Slack message"

        assert [request.kwargs["params"] for request in mock_get.call_args_list] == [
            {"apps": "github"},
            {"apps": "slack"},
        ]

    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_invalid_parameter_schemas_do_not_abort_discovery(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create a GitHub issue",
                        "parameters": "invalid",
                    },
                    {
                        "name": "close_issue",
                        "description": "Close a GitHub issue",
                        "parameters": [{"type": "string"}],
                    },
                    {
                        "name": "list_issues",
                        "description": "List GitHub issues",
                        "parameters": {},
                    },
                ]
            }
        }
        mock_get.return_value = mock_response

        tools = CrewaiPlatformTools(apps=["github"])

        assert [tool.name for tool in tools] == [
            "github_create_issue",
            "github_close_issue",
            "github_list_issues",
        ]
        assert all(tool.args_schema.model_fields == {} for tool in tools)

    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    def test_crewai_platform_tools_empty_apps(self):
        with patch(
            "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"actions": {}}
            mock_get.return_value = mock_response

            tools = CrewaiPlatformTools(apps=[])
            assert tools is not None
            assert isinstance(tools, list)
            assert len(tools) == 0

    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_crewai_platform_tools_api_error_handling(self, mock_get):
        mock_get.side_effect = Exception("API Error")

        with self.assertRaisesRegex(Exception, "API Error"):
            CrewaiPlatformTools(apps=["github"])

    def test_crewai_platform_tools_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                CrewaiPlatformTools(apps=["github"])
            assert "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN" in str(context.exception)

    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_discovered_tool_executes_through_legacy_api(self, mock_get, mock_post):
        discovery_response = Mock()
        discovery_response.raise_for_status.return_value = None
        discovery_response.json.return_value = {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create a GitHub issue",
                        "parameters": {
                            "type": "object",
                            "properties": {"title": {"type": "string"}},
                            "required": ["title"],
                        },
                    }
                ]
            }
        }
        mock_get.return_value = discovery_response
        execution_response = Mock(ok=True, status_code=200)
        execution_response.json.return_value = {"issue": 42}
        mock_post.return_value = execution_response

        tools = CrewaiPlatformTools(apps=["github"])
        result = tools[0].run(title="Contract test")

        assert mock_get.call_args.kwargs["params"] == {"apps": "github"}
        assert mock_post.call_args.kwargs["url"].endswith(
            "/actions/create_issue/execute"
        )
        assert mock_post.call_args.kwargs["json"] == {
            "integration": {"title": "Contract test"}
        }
        assert result == '{\n  "issue": 42\n}'

    @patch.dict(
        "os.environ",
        {
            "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token",
            "CREWAI_PLUS_URL": "https://platform.example.test/",
        },
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post"
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_connection_selects_clipper_api(self, mock_get, mock_post):
        discovery_response = Mock()
        discovery_response.raise_for_status.return_value = None
        discovery_response.json.return_value = {
            "data": {
                "slug": "create_issue",
                "description": "Create a GitHub issue",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            }
        }
        mock_get.return_value = discovery_response
        execution_response = Mock(status_code=200)
        execution_response.json.return_value = {"data": {"output": {"issue": 42}}}
        mock_post.return_value = execution_response
        connection_id = "550e8400-e29b-41d4-a716-446655440000"

        tools = CrewaiPlatformTools(
            apps=[f"github/create_issue@{connection_id}"]
        )
        result = tools[0].run(title="Contract test")

        assert mock_get.call_args.args[0].endswith(
            "/clipper/v1/applications/github/tools/create_issue"
        )
        assert mock_get.call_args.kwargs["params"] == {
            "connection_id": connection_id
        }
        assert mock_post.call_args.args[0].endswith(
            "/clipper/v1/applications/github/tools/create_issue/execute"
        )
        assert mock_post.call_args.kwargs["json"] == {
            "arguments": {"title": "Contract test"},
            "connection_id": connection_id,
        }
        assert result == '{\n  "issue": 42\n}'

    @patch.dict(
        "os.environ",
        {
            "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token",
            "CREWAI_PLUS_URL": "https://platform.example.test/",
        },
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_private_connection_selects_clipper_api(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": []}
        mock_get.return_value = response

        tools = CrewaiPlatformTools(apps=["github@private"])

        assert tools == []
        assert mock_get.call_args.args[0].endswith(
            "/clipper/v1/applications/github/tools"
        )
        assert mock_get.call_args.kwargs["params"] == {"connection_id": "private"}

    @patch.dict(
        "os.environ",
        {
            "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token",
            "CREWAI_PLUS_URL": "https://platform.example.test/",
        },
        clear=True,
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_mixed_selectors_use_both_apis(self, mock_get):
        legacy_response = Mock()
        legacy_response.raise_for_status.return_value = None
        legacy_response.json.return_value = {"actions": {"slack": []}}
        clipper_response = Mock()
        clipper_response.raise_for_status.return_value = None
        clipper_response.json.return_value = {"data": []}
        mock_get.side_effect = [legacy_response, clipper_response]
        connection_id = "550e8400-e29b-41d4-a716-446655440000"

        tools = CrewaiPlatformTools(apps=["slack", f"github@{connection_id}"])

        assert tools == []
        assert mock_get.call_count == 2
        assert mock_get.call_args_list[0].kwargs["params"] == {"apps": "slack"}
        assert mock_get.call_args_list[1].args[0].endswith(
            "/clipper/v1/applications/github/tools"
        )
        assert mock_get.call_args_list[1].kwargs["params"] == {
            "connection_id": connection_id
        }

    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_same_action_from_different_apps_has_unique_tool_names(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "actions": {
                "github": [
                    {
                        "name": "search",
                        "description": "Search GitHub",
                        "parameters": {},
                    }
                ],
                "slack": [
                    {
                        "name": "search",
                        "description": "Search Slack",
                        "parameters": {},
                    }
                ],
            }
        }
        github_response = Mock()
        github_response.raise_for_status.return_value = None
        github_response.json.return_value = {
            "actions": {"github": response.json.return_value["actions"]["github"]}
        }
        slack_response = Mock()
        slack_response.raise_for_status.return_value = None
        slack_response.json.return_value = {
            "actions": {"slack": response.json.return_value["actions"]["slack"]}
        }
        mock_get.side_effect = [github_response, slack_response]

        tools = CrewaiPlatformTools(apps=["github", "slack"])

        assert len(tools) == 2
        assert [tool.name for tool in tools] == ["github_search", "slack_search"]
        assert [tool.app for tool in tools] == ["github", "slack"]
        assert [tool.description for tool in tools] == ["Search GitHub", "Search Slack"]

    @patch.dict("os.environ", {"CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_tool_name_uses_its_sanitized_identity(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": {
                "slug": "CreateFile!",
                "description": "Create a file",
                "input_schema": {},
            }
        }
        mock_get.return_value = response

        tools = CrewaiPlatformTools(
            apps=[
                "Google Drive/CreateFile!@550e8400-e29b-41d4-a716-446655440000"
            ]
        )

        assert tools[0].name == (
            "google_drive_create_file_550e8400_e29b_41d4_a716_446655440000"
        )


_CLIENT = "crewai_tools.tools.crewai_platform_tools.integrations_client"
_CONNECTION = "550e8400-e29b-41d4-a716-446655440000"
_SEND_MESSAGE = {
    "name": "send_message",
    "description": "Send a Slack message",
    "parameters": {
        "type": "object",
        "properties": {
            "channel": {"type": "string"},
            "text": {"type": "string"},
            "thread": {"type": "string"},
            "selector": {"type": "string"},
        },
        "required": ["channel", "text"],
    },
}
_CREATE_ISSUE = {
    "data": {
        "slug": "create_issue",
        "description": "Create a GitHub issue",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
        },
    }
}


def _legacy_discovery(*actions: dict) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"actions": {"slack": list(actions)}}
    return response


def _response(status_code: int, body: object) -> Mock:
    response = Mock(status_code=status_code)
    response.raise_for_status.return_value = None
    response.json.return_value = body
    return response


@pytest.fixture
def platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", "test_token")
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://platform.example.test/")


@pytest.fixture
def mock_get():
    with patch(f"{_CLIENT}.requests.get") as get:
        yield get


@pytest.fixture
def mock_post():
    with patch(f"{_CLIENT}.requests.post") as post:
        yield post


@pytest.mark.usefixtures("platform_env")
class TestRunPlatformAction:
    def test_is_exported_from_the_package_root(self) -> None:
        import crewai_tools
        from crewai_tools.tools.crewai_platform_tools import crewai_platform_tools

        assert (
            crewai_tools.run_platform_action
            is crewai_platform_tools.run_platform_action
        )

    def test_legacy_action_returns_its_output(self, mock_get, mock_post) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)
        mock_post.return_value = _response(200, {"ok": True, "ts": "1.2"})

        output = run_platform_action(
            "slack/send_message", channel="#general", text="hi"
        )

        assert output == {"ok": True, "ts": "1.2"}
        assert mock_get.call_args.kwargs["params"] == {"apps": "slack/send_message"}
        assert mock_post.call_args.kwargs["url"].endswith(
            "/actions/send_message/execute"
        )
        assert mock_post.call_args.kwargs["json"] == {
            "integration": {"channel": "#general", "text": "hi"}
        }

    def test_connection_action_returns_its_output(self, mock_get, mock_post) -> None:
        mock_get.return_value = _response(200, _CREATE_ISSUE)
        mock_post.return_value = _response(200, {"data": {"output": {"issue": 42}}})

        output = run_platform_action(
            f"github/create_issue@{_CONNECTION}", title="Contract test"
        )

        assert output == {"issue": 42}
        assert mock_post.call_args.args[0].endswith(
            "/clipper/v1/applications/github/tools/create_issue/execute"
        )
        assert mock_post.call_args.kwargs["json"] == {
            "arguments": {"title": "Contract test"},
            "connection_id": _CONNECTION,
        }

    def test_server_error_raises_a_retryable_failure(self, mock_get, mock_post) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)
        mock_post.return_value = _response(
            503, {"error": {"message": "Slack is unavailable"}}
        )

        with pytest.raises(
            ToolExecutionFailedError,
            match="API request failed: Slack is unavailable",
        ) as raised:
            run_platform_action("slack/send_message", channel="#general", text="hi")

        record = raised.value.record
        assert record.tool_name == "slack_send_message"
        assert record.tool_args == {"channel": "#general", "text": "hi"}
        assert record.failure.code == "503"
        assert record.failure.retryable is True

    def test_client_error_raises_a_non_retryable_failure(
        self, mock_get, mock_post
    ) -> None:
        mock_get.return_value = _response(200, _CREATE_ISSUE)
        mock_post.return_value = _response(
            422,
            {"errors": [{"detail": "Repository is archived", "code": "archived"}]},
        )

        with pytest.raises(
            ToolExecutionFailedError, match="Repository is archived"
        ) as raised:
            run_platform_action(f"github/create_issue@{_CONNECTION}", title="x")

        assert raised.value.record.failure.code == "archived"
        assert raised.value.record.failure.retryable is False

    def test_execute_exception_raises_a_failure(self, mock_get, mock_post) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)
        mock_post.side_effect = ConnectionError("connection reset")

        with pytest.raises(
            ToolExecutionFailedError,
            match="Error executing action send_message: connection reset",
        ) as raised:
            run_platform_action("slack/send_message", channel="#general", text="hi")

        assert raised.value.record.failure.code == "ConnectionError"

    def test_unknown_action_raises_without_executing(
        self, mock_get, mock_post
    ) -> None:
        mock_get.return_value = _legacy_discovery()

        with pytest.raises(
            ValueError,
            match=(
                "selector 'slack/missing' resolved to 0 actions; the action does "
                "not exist or the app is not connected"
            ),
        ):
            run_platform_action("slack/missing")

        mock_post.assert_not_called()

    def test_several_resolved_actions_raise_without_executing(
        self, mock_get, mock_post
    ) -> None:
        mock_get.return_value = _legacy_discovery(
            _SEND_MESSAGE, {**_SEND_MESSAGE, "name": "send_dm"}
        )

        with pytest.raises(ValueError, match="resolved to 2 actions"):
            run_platform_action("slack/send_message", channel="#general", text="hi")

        mock_post.assert_not_called()

    def test_app_selector_raises_before_discovery(self, mock_get, mock_post) -> None:
        with pytest.raises(
            ValueError,
            match=(
                r"selector 'slack' names an app, not one action; expected "
                r"'<app>/<action>\[@<connection>\]'"
            ),
        ):
            run_platform_action("slack")

        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_malformed_selector_raises_before_discovery(
        self, mock_get, mock_post
    ) -> None:
        with pytest.raises(ValueError, match="action cannot be empty"):
            run_platform_action("slack/")

        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_arguments_that_fail_the_schema_raise_without_executing(
        self, mock_get, mock_post
    ) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)

        with pytest.raises(
            ValueError,
            match="Tool 'slack_send_message' arguments validation failed",
        ):
            run_platform_action("slack/send_message", channel="#general")

        mock_post.assert_not_called()

    def test_argument_named_selector_reaches_the_action(
        self, mock_get, mock_post
    ) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)
        mock_post.return_value = _response(200, {"ok": True})

        run_platform_action(
            "slack/send_message", channel="#general", text="hi", selector="mine"
        )

        assert mock_post.call_args.kwargs["json"] == {
            "integration": {"channel": "#general", "text": "hi", "selector": "mine"}
        }

    def test_none_arguments_are_not_sent(self, mock_get, mock_post) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)
        mock_post.return_value = _response(200, {"ok": True})

        run_platform_action(
            "slack/send_message", channel="#general", text="hi", thread=None
        )

        assert mock_post.call_args.kwargs["json"] == {
            "integration": {"channel": "#general", "text": "hi"}
        }

    def test_output_that_is_not_an_object_raises(self, mock_get, mock_post) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)
        mock_post.return_value = _response(200, ["not", "an", "object"])

        with pytest.raises(
            TypeError,
            match="action 'slack/send_message' returned list, not a JSON object",
        ):
            run_platform_action("slack/send_message", channel="#general", text="hi")

    def test_missing_token_raises_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch, mock_get, mock_post
    ) -> None:
        monkeypatch.delenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN")

        with pytest.raises(
            ValueError,
            match="CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN environment variable",
        ):
            run_platform_action("slack/send_message", channel="#general", text="hi")

        mock_get.assert_not_called()
        mock_post.assert_not_called()

    def test_discovered_tool_run_still_returns_a_failure(
        self, mock_get, mock_post
    ) -> None:
        mock_get.return_value = _legacy_discovery(_SEND_MESSAGE)
        mock_post.return_value = _response(
            503, {"error": {"message": "Slack is unavailable"}}
        )

        [tool] = CrewaiPlatformTools(apps=["slack/send_message"])
        result = tool.run(channel="#general", text="hi")

        assert isinstance(result, ToolFailure)
        assert result.message == "API request failed: Slack is unavailable"
