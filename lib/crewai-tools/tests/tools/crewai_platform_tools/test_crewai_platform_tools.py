import unittest
from unittest.mock import Mock, patch

from crewai_tools.tools.crewai_platform_tools import CrewaiPlatformTools


class TestCrewaiPlatformTools(unittest.TestCase):
    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
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
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
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

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert (
            "apps=github,slack" in args[0]
            or kwargs.get("params", {}).get("apps") == "github,slack"
        )

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    def test_crewai_platform_tools_empty_apps(self):
        with patch(
            "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
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
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
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
