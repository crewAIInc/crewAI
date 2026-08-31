import unittest
from unittest.mock import Mock, patch

import requests

from crewai_tools.tools.crewai_platform_tools import CrewaiPlatformTools


class TestCrewaiPlatformTools(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "CREWAI_PLATFORM_INTEGRATION_TOKEN": "token",
            "CREWAI_PLUS_URL": "https://platform.example",
        },
    )
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
    )
    def test_uses_clipper_for_selector_with_connection_id(self, mock_get):
        response = Mock()
        response.json.return_value = {
            "data": {
                "slug": "create-issue",
                "description": "Create a GitHub issue",
                "input_schema": {"type": "object", "properties": {}},
            }
        }
        mock_get.return_value = response

        tools = CrewaiPlatformTools(
            apps=[
                "github/create-issue@550e8400-e29b-41d4-a716-446655440000",
            ]
        )

        assert [tool.name for tool in tools] == ["github_create_issue"]
        assert mock_get.call_args.args[0] == (
            "https://platform.example/clipper/v1/applications/github/tools/create-issue"
        )

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
    )
    def test_crewai_platform_tools_basic(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {"github": []}}
        mock_get.return_value = mock_response

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert isinstance(tools, list)

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
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
        actions = mock_response.json.return_value["actions"]

        def response_for_app(*args, **kwargs):
            app = kwargs["params"]["apps"]
            response = Mock()
            response.json.return_value = {"actions": {app: actions[app]}}
            return response

        mock_get.side_effect = response_for_app

        tools = CrewaiPlatformTools(apps=["github", "slack"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 2

        assert mock_get.call_count == 2
        assert [call.kwargs["params"]["apps"] for call in mock_get.call_args_list] == [
            "github",
            "slack",
        ]

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    def test_crewai_platform_tools_empty_apps(self):
        with patch(
            "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
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
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.get"
    )
    def test_crewai_platform_tools_api_error_handling(self, mock_get):
        mock_get.side_effect = requests.RequestException("API Error")

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_crewai_platform_tools_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                CrewaiPlatformTools(apps=["github"])
            assert "No platform integration token found" in str(context.exception)

    def test_rejects_invalid_selectors(self):
        cases = [
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
        ]

        for selector, message in cases:
            with self.subTest(selector=selector):
                with self.assertRaisesRegex(ValueError, message):
                    CrewaiPlatformTools(apps=[selector])
