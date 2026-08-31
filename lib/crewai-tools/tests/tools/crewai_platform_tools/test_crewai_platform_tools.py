import unittest
from unittest.mock import Mock, patch

import requests

from crewai_tools.tools.crewai_platform_tools import CrewaiPlatformTools


class TestCrewaiPlatformTools(unittest.TestCase):
    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools._client.requests.get"
    )
    def test_crewai_platform_tools_basic(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert isinstance(tools, list)

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools._client.requests.get"
    )
    def test_crewai_platform_tools_multiple_apps(self, mock_get):
        github_response = Mock()
        github_response.raise_for_status.return_value = None
        github_response.json.return_value = {
            "data": [
                    {
                        "slug": "create_issue",
                        "description": "Create a GitHub issue",
                        "input_schema": {
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
                ]
        }
        slack_response = Mock()
        slack_response.raise_for_status.return_value = None
        slack_response.json.return_value = {
            "data": [
                    {
                        "slug": "send_message",
                        "description": "Send a Slack message",
                        "input_schema": {
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
                ]
        }
        mock_get.side_effect = [github_response, slack_response]

        tools = CrewaiPlatformTools(apps=["github", "slack"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 2

        assert [call.args[0].split("/clipper", 1)[1] for call in mock_get.call_args_list] == [
            "/v1/applications/github/tools",
            "/v1/applications/slack/tools",
        ]

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
    def test_keeps_same_named_actions_from_different_apps(self, mock_get):
        github_response = Mock()
        github_response.raise_for_status.return_value = None
        github_response.json.return_value = {
            "data": [{"slug": "search", "description": "Search", "input_schema": {}}]
        }
        slack_response = Mock()
        slack_response.raise_for_status.return_value = None
        slack_response.json.return_value = {
            "data": [{"slug": "search", "description": "Search", "input_schema": {}}]
        }
        mock_get.side_effect = [github_response, slack_response]

        tools = CrewaiPlatformTools(apps=["github", "slack"])

        assert tools["github_search"].app == "github"
        assert tools["slack_search"].app == "slack"

    @patch.dict(
        "os.environ",
        {
            "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
            "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment_uuid",
        },
        clear=True,
    )
    @patch("crewai_tools.tools.crewai_platform_tools._client.requests.post")
    @patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
    def test_executes_same_action_for_each_app(self, mock_get, mock_post):
        github_response = Mock()
        github_response.raise_for_status.return_value = None
        github_response.json.return_value = {
            "data": {
                "slug": "search",
                "description": "Search",
                "input_schema": {
                    "properties": {"query": {"type": "string"}}
                },
            }
        }
        slack_response = Mock()
        slack_response.raise_for_status.return_value = None
        slack_response.json.return_value = github_response.json.return_value
        mock_get.side_effect = [github_response, slack_response]
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"data": {"output": []}}
        mock_post.return_value = response

        tools = CrewaiPlatformTools(apps=["github/search", "slack/search"])
        tools["github_search"].run(query="release")
        tools["slack_search"].run(query="release")

        assert [call.kwargs["url"] for call in mock_post.call_args_list] == [
            "https://app.crewai.com/clipper/v1/applications/github/tools/search/execute",
            "https://app.crewai.com/clipper/v1/applications/slack/tools/search/execute",
        ]

    @patch.dict(
        "os.environ",
        {
            "CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token",
            "CREWAI_DEPLOYMENT_INSTANCE_UUID": "deployment_uuid",
        },
        clear=True,
    )
    @patch("crewai_tools.tools.crewai_platform_tools._client.requests.post")
    @patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
    def test_executes_same_action_for_each_connection(self, mock_get, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": {
                "slug": "search",
                "description": "Search",
                "input_schema": {
                    "properties": {"query": {"type": "string"}}
                },
            }
        }
        mock_get.return_value = response
        execution_response = Mock(ok=True, status_code=200)
        execution_response.json.return_value = {"data": {"output": []}}
        mock_post.return_value = execution_response
        connection_ids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "67e55044-10b1-426f-9247-bb680e5fe0c8",
        ]

        tools = CrewaiPlatformTools(
            apps=[f"github/search@{connection_id}" for connection_id in connection_ids]
        )
        for tool in tools:
            tool.run(query="release")

        assert len(tools) == 2
        assert len({tool.name for tool in tools}) == 2
        assert all(tools[tool.name] is tool for tool in tools)
        assert [call.kwargs["json"] for call in mock_post.call_args_list] == [
            {
                "arguments": {"query": "release"},
                "connection_id": connection_id,
            }
            for connection_id in connection_ids
        ]

    @patch.dict(
        "os.environ",
        {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"},
        clear=True,
    )
    @patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
    def test_invalid_discovery_response_returns_empty_collection(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = response

        tools = CrewaiPlatformTools(apps=["github"])

        assert tools == []

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    def test_crewai_platform_tools_empty_apps(self):
        with patch(
            "crewai_tools.tools.crewai_platform_tools._client.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            tools = CrewaiPlatformTools(apps=[])
            assert tools is not None
            assert isinstance(tools, list)
            assert len(tools) == 0

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools._client.requests.get"
    )
    def test_crewai_platform_tools_api_error_handling(self, mock_get):
        mock_get.side_effect = requests.RequestException("API Error")

        tools = CrewaiPlatformTools(apps=["github"])
        assert tools is not None
        assert isinstance(tools, list)
        assert len(tools) == 0

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch("crewai_tools.tools.crewai_platform_tools._client.requests.get")
    def test_crewai_platform_tools_handles_unexpected_fetch_error(self, mock_get):
        mock_get.side_effect = RuntimeError("API Error")

        tools = CrewaiPlatformTools(apps=["github"])

        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_crewai_platform_tools_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                CrewaiPlatformTools(apps=["github"])
            assert "No platform integration token found" in str(context.exception)
