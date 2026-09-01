import unittest
from unittest.mock import Mock, patch

from crewai_tools.tools.crewai_platform_tools import CrewaiPlatformTools


class TestCrewaiPlatformTools(unittest.TestCase):
    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
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

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
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
        mock_get.return_value = mock_response

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

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
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

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
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

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.integrations_client.requests.get"
    )
    def test_crewai_platform_tools_api_error_handling(self, mock_get):
        mock_get.side_effect = Exception("API Error")

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_crewai_platform_tools_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                CrewaiPlatformTools(apps=["github"])
            assert "No platform integration token found" in str(context.exception)

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
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
            "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
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
            "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
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

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
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
        mock_get.return_value = response

        tools = CrewaiPlatformTools(apps=["github", "slack"])

        assert len(tools) == 2
        assert [tool.name for tool in tools] == ["github_search", "slack_search"]
        assert [tool.app for tool in tools] == ["github", "slack"]
        assert [tool.description for tool in tools] == ["Search GitHub", "Search Slack"]

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
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
